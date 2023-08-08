from fenics import *
import tracerdiffusion.config as config

if config.inverse:
    print("config.inverse = True, importing dolfin-adjoint")
    from fenics_adjoint import *
else:
    print("config.inverse = False, not importing dolfin-adjoint")

import pathlib
import numpy as np
import json
from tqdm import tqdm


class Model(object):
    def __init__(self, dt, V, mris, outfolder: pathlib.Path, dx_SD=None, verbosity: int=0):
        """
        :param mesh_config: dictionary which contains ds, dx
        :param V: FunctionSpace for state variable
        :param delta: diffusion coefficient
        :param params: dictionary which contains time step size self.dt
        :param data: class
        """

        mean_diffusivity_water = 3e-3
        mean_diffusivity_gadovist = 3.8e-4 # (taken from Gd-DPTA)
        scale_diffusion_gad = mean_diffusivity_gadovist / mean_diffusivity_water
        self.scale_diffusion_gad = scale_diffusion_gad
        self.D_scale = Constant(scale_diffusion_gad) 
                

        self.V = V

        self.ds = Measure('ds')(domain=self.V.mesh())
        #  ds # mesh_config["ds"]
        self.dx = Measure('dx')(domain=self.V.mesh()) # mesh_config["dx"]

        self.dx_SD = dx_SD

        self.data = mris.tolist()

        self.mri_times = [x * 3600 for x in mris.measurement_times()] # simulation scripts work in units of hours
        self.T = max(self.mri_times)

        assert max(mris.measurement_times()) <= 6 * 24, "MRI times should be in hours, maybe something is wrong?"

        self.verbosity = verbosity
        
        if not isinstance(dt, Constant):
            self.timestep = Constant(dt)

        self.dt = dt

        print("Simulating for", format(self.T, ".0f"), "hours",
                "(", format(self.T / 1, ".0f"), "seconds)"
                " with time step size", format(self.dt / 60, ".0f"), "minutes")

        # self.linear_solver_args = ("gmres", "amg")
        
        self.outfolder = outfolder
        self.brain_volume = assemble(Constant(1) * self.dx)
        self.brain_surface_area = assemble(1 * self.ds)
        
        if self.dx_SD is not None:
            self.gray_volume = assemble(1 * dx_SD(1))
            self.white_matter_volume = assemble(1 * dx_SD(2))
            self.brain_stem_volume = assemble(1 * dx_SD(3))

        self.concfile = self.outfolder / 'simulated_concentration.txt'
        with open(self.concfile, 'w') as file:
            file.write('t avg avgds ')
            if self.dx_SD is not None:
                file.write('gray white brainstem ')
            file.write('\n')
     
        
        
        times = [0]
        t = 0
        while t + self.dt <= self.T:
            t += self.dt
            times.append(t)
            

        times = np.array(times)

        self.checkpoints = []
        for mri_time in self.mri_times:
            print("Data available at", format(mri_time / 3600, ".0f"), "hours past first image")
            self.checkpoints.append(np.round(times[np.argmin(np.abs(times - mri_time))], 0).item())


    def save_predictions(self, name="simulation"):


        pvdfile = File(str(self.outfolder / (name + ".pvd")))
        
        for sim in self.simulated_tracer:
            sim.rename("simulation", "simulation")
            pvdfile << sim

    
        hdf5file = HDF5File(self.V.mesh().mpi_comm(), str(self.outfolder / (name + ".hdf")), "w")
        hdf5file.write(self.V.mesh(), "mesh")
        
        for idx, sim in enumerate(self.simulated_tracer):
            sim.rename(name, name)
            hdf5file.write(sim, name + format(self.checkpoints[idx]/3600, ".0f"))

        hdf5file.close()

        checkpoints = {"simulation": self.checkpoints,
                        "data": self.mri_times}

        with open(self.outfolder / "checkpoints.json", 'w') as outfile:
            json.dump(checkpoints, outfile, sort_keys=True, indent=4)


    def advance_time(self, current_state):
        """
        update of time variables and boundary conditions
        """
        self.t += self.dt     # update time-step



        if self.t > self.mri_times[self.image_counter]:

            self.image_counter_prev = self.image_counter
            
            while self.t > self.mri_times[self.image_counter]:

                if self.verbosity == 1:
                    print("t=", format(self.t / 3600, ".2f"), "Increasing image counter from", self.image_counter, "to", self.image_counter + 1)
                
                self.image_counter += 1


                
        # if not self.next_image_index == len(self.times) and 
        if np.round(self.t, 0) in self.checkpoints:

            dt = self.t - self.mri_times[np.argmin(np.abs(np.array(self.checkpoints)-self.t))]

            if dt > 0:
                # In this case our current time is a bit over the imaging time. 
                # To compute the error w.r.t. to the nearest image we use the previous counter:
                i = self.image_counter - 1
            else:
                i = self.image_counter


            if self.verbosity == 1:
                print("Computing L2 error at t=", format(self.t / 3600, ".2f"), "(image ", i + 1, "/", len(self.data), ")")

            if config.inverse:
                self.simulated_tracer.append(current_state.copy(deepcopy=True, annotate=False))
            else:
                self.simulated_tracer.append(current_state.copy(deepcopy=True))

            L2_error = assemble((current_state - self.data[i]) ** 2 * self.dx) 

            datanorm = assemble((self.data[i]) ** 2 * self.dx)
            
            self.datanorm += datanorm
            
            if self.verbosity == 0:
                print("Rel. L2 error ||c-cdata|| / ||cdata|| at t=", format(self.t / 3600, ".2f"), "is", format(L2_error / datanorm, ".2f"), "(image ", i + 1, "/", len(self.data), ")")

            assert L2_error < 1e14

            self.L2_error += L2_error


    def boundary_condition(self):
        """
        Linear interpolation c_1 + (c2 - c1) / (t2 - t1) * (t - t1) of image data as boundary condition
        """

        def boundary(x, on_boundary):
            return on_boundary
        
        # return DirichletBC(self.V, Constant(self.t / self.T), boundary)
    
        try:
            return self.bcs[self.t]
        
        except KeyError:

            if self.verbosity == 1:
                print("time=", format(self.t / 3600, ".0f"), "h, image_counter=", self.image_counter)
                            
            self.linear_interpolation.vector()[:] = self.data[self.image_counter_prev].vector()[:]
            
            finite_difference = (self.data[self.image_counter].vector()[:] - self.data[self.image_counter_prev].vector()[:])
            finite_difference /= (self.mri_times[self.image_counter] - self.mri_times[self.image_counter_prev])

            self.linear_interpolation.vector()[:] +=  finite_difference * (self.t - self.mri_times[self.image_counter_prev])

            if config.inverse:
                self.bcs[self.t] = DirichletBC(self.V, self.linear_interpolation.copy(deepcopy=True, annotate=False), boundary)
            else:
                self.bcs[self.t] = DirichletBC(self.V, self.linear_interpolation.copy(deepcopy=True), boundary)
                                         
            return self.bcs[self.t]

    def store_values(self, fun):

        with open(self.concfile, 'a') as file:

            file.write('%g ' % self.t)
            average = assemble(fun * self.dx) / self.brain_volume
            average_ds = assemble(fun * self.ds) / self.brain_surface_area
            
            if self.dx_SD is not None:
                gray_avg = assemble(fun * self.dx_SD(1)) / self.gray_volume
                white_avg = assemble(fun * self.dx_SD(2)) / self.white_matter_volume
                brain_stem_avg = assemble(fun * self.dx_SD(3)) / self.brain_stem_volume

            file.write('%g ' % average)
            file.write('%g ' % average_ds)
            
            if self.dx_SD is not None:
                file.write('%g ' % gray_avg)
                file.write('%g ' % white_avg)
                file.write('%g ' % brain_stem_avg)
            file.write('\n')



    def return_value(self):
        return self.L2_error


    def forward(self, water_diffusivity, diffusion_tensor_water=None, r=None, taylortest=False):
        self.L2_error = 0.0
        self.datanorm = 0.0
        self.bcs = {}
        self.t = 0
        self.state_at_measurement_points = []
        self.image_counter = 1
        self.image_counter_prev = 0
        self.simulated_tracer = []
        self.linear_interpolation = Function(self.V)
        
        assert diffusion_tensor_water is None
        diffusion_tensor = None
        
        D = self.D_scale * water_diffusivity
        
        # Workaround to get the value of the FEniCS constant back:
        vals = np.zeros(1)
        water_diffusivity.eval(vals, np.zeros(1))
        print("Using D = " + format((self.scale_diffusion_gad * vals)[0], ".2e") + " mm^2/s for gadobutrol diffusion (cf. Valnes et al Scientific Reports 2020 for details)")

        # Define trial and test-functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Solution at current and previous time
        u_prev = Function(self.V)
        u_next = Function(self.V)

        u_prev.assign(self.data[0])
        u_next.assign(self.data[0])

        pvdfile = File(str(self.outfolder / "movie.pvd"))
        # u_prev.rename("simulation", "simulation    ")
        # pvdfile << u_prev


        movie_hdf = HDF5File(self.V.mesh().mpi_comm(), str(self.outfolder / "movie.hdf"), "w")
        movie_hdf.write(self.V.mesh(), "mesh")
        
        if config.inverse:
            self.simulated_tracer.append(u_prev.copy(deepcopy=True, annotate=False))
        else:
            self.simulated_tracer.append(u_prev.copy(deepcopy=True))
        
        try:
            self.store_values(fun=u_prev)
        except:
            pass

        iter_k = 0

        def diffusion(fun):

            if self.dx_SD is not None:
                raise NotImplementedError

                # gray matter:
                term = D * inner(grad(fun), grad(v)) * self.dx_SD(1)
                
                if diffusion_tensor is not None:
                    # use DTI in white matter
                    term += inner(dot(diffusion_tensor, grad(fun)), grad(v)) * self.dx_SD(2)
                else:
                    # use mean diffusivity in white matter
                    term += self.mean_diffusivity * inner(grad(fun), grad(v)) * self.dx_SD(2)
                
                # brain stem:
                term += self.mean_diffusivity * inner(grad(fun), grad(v)) * self.dx_SD(3)
            
            # no subdomains
            else:
                term = D * inner(grad(fun), grad(v)) * self.dx

            return term

        def reaction(fun):
            return r * inner(fun, v) * self.dx

        a = inner(u, v) * self.dx
        # NOTE: if you change this to explicit/Crank-Nicolson, change L in the loop!
        # implicit time stepping:

        a += self.timestep * diffusion(fun=u)
        
        if r is not None:
            a += self.timestep * reaction(fun=u)

        A = assemble(a)

        # breakpoint()
        if taylortest:
            solver = LUSolver()
            solver.set_operator(A)
        else:
            solver = PETScKrylovSolver('gmres', 'amg')
            solver.set_operators(A, A)

        solver = LUSolver()
        solver.set_operator(A)

        if self.verbosity == 0:
            progress = tqdm(total=int(self.T / self.dt))


        while self.t + self.dt / 1 <= self.T:
            
            iter_k += 1

            u_prev.rename("simulation", "simulation")
            pvdfile << u_prev
            movie_hdf.write(u_prev, format(self.t / 3600, ".1f"))

            u_prev.assign(u_next)
            
            self.advance_time(u_prev)
            
            # get current BC:
            bc = self.boundary_condition()
            # continue
            bc.apply(A)

            # Assemble RHS and apply DirichletBC
            rhs = u_prev * v * self.dx
            b = assemble(rhs)
            bc.apply(b)

            # Solve A* u_current = b
            solver.solve(u_next.vector(), b)

            # solve(A, U.vector(), b, 'lu')
            try:
            
                self.store_values(fun=u_next)
            except:
                pass

            if self.verbosity == 0:
                progress.update(1)

        

        u_prev.assign(u_next)
        u_prev.rename("simulation", "simulation")
        pvdfile << u_prev

        movie_hdf.write(u_prev, format(self.t / 3600, ".1f"))

        print("Done with simulation")

        return self.return_value()