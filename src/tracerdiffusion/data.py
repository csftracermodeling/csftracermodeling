try:
    from dolfin import *
    import tracerdiffusion.config as config

    if config.inverse:
        from dolfin_adjoint import *
    # import h5py
except ModuleNotFoundError:
    print("Could not import dolfin and/or dolfin-adjoint")

import itertools
from tracerdiffusion.domains import ImageDomain
import pathlib
import nibabel
import numpy
from nibabel.affines import apply_affine
import os
from datetime import datetime
import abc
from parse import parse

TMAX_HOURS = 4 * 24 * 60 * 60


def get_delta_t(file1: str, file2: str, file_suffix="", name_format=None) -> float:
    """
    TODO complete docstring

    Assumes files to be named as name_format + file_suffix + ".type",

    Args:
        file1 (str): filename of first image, without .mgz
        file2 (str): filename of second image, without .mgz
        file_suffix (str, optional): _description_. Defaults to "".
        name_format (str, optional): _description_. Defaults to '%Y%m%d_%H%M%S'.

    Raises:
        ValueError: _description_

    Returns:
        float: time in hours between two images
    """

    dates = []

    if name_format is not None:
        assert name_format == "%Y%m%d_%H%M%S"

        for file in [file1, file2]:
            file = pathlib.Path(file).stem

            if "/" in file or "." in file:
                raise ValueError("Expecting string of format " + name_format)

            date_time = file.replace(file_suffix, "")

            dates.append(datetime.strptime(date_time, name_format))

        difference = dates[1] - dates[0]
        time = difference.days * 3600 * 24 + difference.seconds

    else:

        def time_from_filename(file):
            t = pathlib.Path(file)

            if t.is_file():
                t = t.stem
            else:
                t = t.name

            parser_result = parse("{}.{}", t)

            if parser_result is None or parser_result[0] is None:
                raise ValueError(
                    "Could not infer time in hours from file name. Expecting file name to be of form <12.34.mgz> or <1.23.mgz>"
                )
            hours = float(parser_result[0])
            minutes = float(parser_result[1])
            assert minutes < 60
            time = hours + minutes / 60

            return time

        time = abs(time_from_filename(file1) - time_from_filename(file2))

    if time > TMAX_HOURS:
        raise ValueError(
            str(time)
            + "> TMAX_HOURS="
            + str(TMAX_HOURS)
            + ", something wrong in file name conversion?"
        )

    return time


class Nan_Filter:

    """
    A filter to replace nan values in a voxel (ijk) with the median value of the adjacent voxels (i+-1, j+-1, k+-1).
    This is needed since the mesh has sub-voxel resolution and hence some mesh vertices might be outside the
    region where we have computed the tracer.

    """

    def __init__(self, mask):
        filter_size = 1

        ajdacent_idx = []

        for x, y, z in itertools.product(
            range(-filter_size, filter_size + 1), repeat=3
        ):
            ajdacent_idx.append([x, y, z])

        # for x in range(- filter_size, filter_size + 1):
        #     for y in range(- filter_size, filter_size + 1):
        #         for z in range(- filter_size, filter_size + 1):
        #             ajdacent_idx.append([x, y, z])

        ajdacent_idx.remove([0, 0, 0])
        self.ajdacent_idx = numpy.array(ajdacent_idx)

        self.mask = mask

    def __call__(self, data, ijk, i, j, k):
        data = numpy.where(self.mask, data, numpy.nan)

        nan_idx = numpy.argwhere(numpy.isnan(data[i, j, k]))

        nan_ijk = ijk[:, nan_idx[:, 0]]
        ni, nj, nk = numpy.rint(nan_ijk).astype("int")

        filtered_data = numpy.copy(data)
        idx = numpy.zeros_like(self.ajdacent_idx)

        for x, y, z in zip(ni, nj, nk):
            idx[:, 0] = self.ajdacent_idx[:, 0] + x
            idx[:, 1] = self.ajdacent_idx[:, 1] + y
            idx[:, 2] = self.ajdacent_idx[:, 2] + z

            sr = data[idx[:, 0], idx[:, 1], idx[:, 2]]

            sr = sr[~numpy.isnan(sr)]

            try:
                assert sr.size > 0
            except AssertionError:
                breakpoint()

            filtered_data[x, y, z] = numpy.median(sr)

        return filtered_data


try:

    def read_image(
        filename: str, functionspace: FunctionSpace, mask: numpy.ndarray = None
    ) -> Function:
        """Read MRI data to FEniCS mesh

        Args:
            filename (str): Path to MRI as .mgz file
            functionspace (FunctionSpace): FEniCS FunctionSpace on which the image will be represented
            mask (numpy.ndarray, optional): Can be used to define a filter that replaces nan voxels with the median value
                                            of adjacent voxels that are in the mask. This is useful because the mesh has
                                            sub-voxel resolution and sometimes mesh vertices might correspond to a voxel
                                            that is not inside the region where the MRI contains valid data.
                                            An example is MRI containing CSF tracer concentration, which sometimes can
                                            only be computed in the brain but not the CSF.
                                            Defaults to None.

        Raises:
            NotImplementedError: Currently only works for 3-D meshes, not slices.

        Returns:
            Function: FEniCS function representing the MRI
        """

        print("Loading", filename)
        if filename.endswith(".mgz"):
            mri_volume = nibabel.load(filename)
            voxeldata = mri_volume.get_fdata()

        if mask is not None:
            if isinstance(mask, str):
                mask = nibabel.load(mask).get_fdata()

            voxeldata = numpy.where(mask, voxeldata, numpy.nan)

            nanfilter = Nan_Filter(mask)

        c_data = Function(functionspace)
        ras2vox_tkr_inv = numpy.linalg.inv(mri_volume.header.get_vox2ras_tkr())

        xyz = functionspace.tabulate_dof_coordinates()

        if functionspace.mesh().topology().dim() == 2:
            raise NotImplementedError()

            assert len(slice_params) > 0

            assert numpy.sum(numpy.abs(slice_params["slice_normal"])) == 1

            xyz2 = numpy.insert(
                xyz,
                numpy.argmax(numpy.abs(slice_params["slice_normal"])),
                -16,  # - slice_params["offset"] - 19.3500, # -17.9447,
                axis=1,
            )

            # # TODO FIXME
            # # why do I need to compute * (-1) ??
            # xyz2[:, -1] *= -1
            # breakpoint()
            # TODO FIXME

        else:
            xyz2 = xyz

        ijk = apply_affine(ras2vox_tkr_inv, xyz2).T
        i, j, k = numpy.rint(ijk).astype("int")

        if mask is not None:
            voxeldata = nanfilter(voxeldata, ijk, i, j, k)

        if numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum() > 0:
            print(
                "Setting",
                numpy.where(numpy.isnan(voxeldata[i, j, k]), 1, 0).sum(),
                "/",
                i.size,
                " nan voxels to 0",
            )
            voxeldata[i, j, k] = numpy.where(
                numpy.isnan(voxeldata[i, j, k]), 0, voxeldata[i, j, k]
            )
        if numpy.where(voxeldata[i, j, k] < 0, 1, 0).sum() > 0:
            print(
                "",
                numpy.where(voxeldata[i, j, k] < 0, 1, 0).sum(),
                "/",
                i.size,
                " voxels in mesh have value < 0, thresholding to 0",
            )
            voxeldata[i, j, k] = numpy.where(
                voxeldata[i, j, k] < 0, 0, voxeldata[i, j, k]
            )

        c_data.vector()[:] = voxeldata[i, j, k]

        # Compute total tracer information, note that factor 1e-6 is needed since concentration is mmol/L and mesh unit is mm
        # print("In", filename, format(numpy.nansum(voxeldata[i, j, k]) * 1e-6,".1e"), "mmol tracer in region defined by mesh" )
        print(
            "In",
            filename,
            format(assemble(c_data * dx) * 1e-6, ".1e"),
            "mmol tracer in region defined by mesh",
        )

        return c_data
except NameError:
    pass


class MRI_Data(abc.ABC):
    def __init__(self, datapath, Tmax=4 * 24, verbosity: int = 1):
        self.datapath = pathlib.Path(datapath)

        if not Tmax < TMAX_HOURS:
            raise ValueError("Time should be in hours")

        self.verbosity = verbosity

        files = sorted(
            [
                self.datapath / x
                for x in os.listdir(self.datapath)
                if "template" not in x
            ],
            key=lambda x: float(pathlib.Path(x).stem),
        )

        self.files = [
            x
            for x in files
            if str(x).endswith(".mgz")
            or str(x).endswith(".nii")
            or str(x).endswith(".nii.gz")
        ]

        self.measurements = {}
        self.time_filename_mapping = {}

        for filename in self.files:
            dt = float(filename.stem)
            # Obsolete version where concentrations were named as YYYYMMDD_HHMMSS.mgz
            # dt = get_delta_t(self.files[0].stem, filename.stem)

            if dt > Tmax:
                if verbosity == 1:
                    print("Omit image", filename)
                    continue

            key = self.timeformat(dt)

            self.time_filename_mapping[key] = filename
            if verbosity == 1:
                print(
                    "Key=",
                    key,
                    "corresponds to image=",
                    self.time_filename_mapping[key],
                    "in data.time_filename_mapping",
                )

        if not max(self.measurement_times()) < TMAX_HOURS:
            raise ValueError("Time should be in hours")

    def tolist(self):
        return [image_function for _, image_function in self.measurements.items()]

    def timeformat(self, t):
        return format(t, ".2f")

    def get_measurement(self, t):
        t = self.timeformat(t)
        return self.measurements[t]

    def measurement_times(self):
        return list(map(lambda x: float(x), list(self.time_filename_mapping.keys())))


class Voxel_Data(MRI_Data):
    def __init__(
        self, mask=None, pixelsizes: list = [1, 1, 1], *args, **kwargs
    ) -> None:
        super(Voxel_Data, self).__init__(*args, **kwargs)

        self.mask = mask

        # if len(self.mask.shape) == 3:
        self.domain = ImageDomain(
            mask=self.mask, pixelsizes=pixelsizes, verbosity=self.verbosity
        )
        self.voxel_center_coordinates = self.domain.voxel_center_coordinates

        self.make_coords()

    def bounds(self) -> tuple:
        minimum = numpy.hstack((min(self.measurement_times()), self.domain.min))

        maximum = numpy.hstack((max(self.measurement_times()), self.domain.max))

        return minimum, maximum

    def make_coords(self):
        self.datatensor = numpy.zeros(
            (
                len(self.time_filename_mapping),
                self.voxel_center_coordinates.shape[0],
                self.voxel_center_coordinates.shape[1] + 2,
            )
        )

        self.datatensor += numpy.nan

        counter = 0

        for key, filename in self.time_filename_mapping.items():
            if str(filename).endswith(".mgz"):
                mri_volume = nibabel.load(filename)
                voxeldata = mri_volume.get_fdata()
            elif str(filename).endswith(".npy"):
                voxeldata = numpy.load(filename)

            # store every input-output-pair as a tuple
            self.measurements[key] = (
                numpy.hstack(
                    (
                        numpy.zeros((self.voxel_center_coordinates.shape[0], 1))
                        + float(key),
                        numpy.copy(self.voxel_center_coordinates),
                    )
                ),
                numpy.expand_dims(voxeldata[self.mask], -1),
            )

            self.datatensor[counter, ...] = numpy.hstack(self.measurements[key])

            counter += 1

        assert len(self.datatensor[numpy.isnan(self.datatensor)]) == 0

    def sample(self, n: int) -> tuple:
        time_samples = numpy.random.randint(
            low=0, high=self.datatensor.shape[0], size=n
        )
        space_samples = numpy.random.randint(
            low=0, high=self.datatensor.shape[1], size=n
        )

        t_xy_c = self.datatensor[time_samples, space_samples, :]

        return t_xy_c[:, :-1], t_xy_c[:, -1]

    def sample_image(self, n: int, time_idx: int) -> tuple:
        space_samples = numpy.random.randint(
            low=0, high=self.datatensor.shape[1], size=n
        )

        t_xy_c = self.datatensor[time_idx, space_samples, :]

        return t_xy_c[:, :-1], t_xy_c[:, -1]


class FEniCS_Data(MRI_Data):
    def __init__(
        self, function_space=None, mask: numpy.ndarray = None, *args, **kwargs
    ) -> None:
        super(FEniCS_Data, self).__init__(*args, **kwargs)

        self.function_space = function_space

        if function_space.mesh().topology().dim() == 2:
            raise NotImplementedError
            assert len(slice_params) > 0

        for key, filename in self.time_filename_mapping.items():
            if mask is None:
                print("Creating mask for the brain from NaNs in the image.")

                # We have set non-brain voxels to NAN, so we can create a mask on-the-fly:
                image = nibabel.load(filename=filename).get_fdata()

                mask = ~numpy.isnan(image)

                print(
                    "Mask covers",
                    format(mask.sum() * 100 / mask.size, ".0f"),
                    r"% of the image voxels.",
                )

                if numpy.allclose(mask.sum() / mask.size, 1) or mask.sum() == 0:
                    raise ValueError(
                        "Mask covers either no or all voxels, did you set all voxel values outside the brain to np.nan ?"
                    )

            self.measurements[key] = read_image(
                str(filename), functionspace=self.function_space, mask=mask
            )

            self.measurements[key].rename("data", "data")

    def dump_pvd(self, vtkpath):
        """Dump all data snapshots to a pvd file

        Args:
            vtkpath (str): Path to export to
        """

        if os.path.isdir(vtkpath):
            vtkpath = vtkpath / "data.pvd"

        vtkfile = File(str(vtkpath))

        for t in self.measurement_times():
            u = self.get_measurement(t)
            vtkfile << u
