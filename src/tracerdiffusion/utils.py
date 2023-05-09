import numpy as np
try:
    from scipy.ndimage import convolve
    from skimage.measure import marching_cubes
    import cc3d
except ModuleNotFoundError:
    print("Some modulels for plotting could not be imported")

import nibabel
from nibabel.affines import apply_affine
from typing import Tuple
import itertools
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass



def function_to_image(function, template_image, extrapolation_value, mask: str = None, allow_full_mask: bool = False) -> Tuple[nibabel.Nifti1Image, np.ndarray]:

    shape = template_image.get_fdata().shape

    output_data = np.zeros(shape) + extrapolation_value

    vox2ras = template_image.header.get_vox2ras_tkr()

    V = function.function_space()

    ## Code to get a bounding box for the mesh, used to not iterate over all the voxels in the image
    if mask is None:
        imap = V.dofmap().index_map()
        num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
        xyz = V.tabulate_dof_coordinates()
        xyz = xyz.reshape((num_dofs_local, -1))
        image_coords = apply_affine(np.linalg.inv(vox2ras), xyz)
        
        lower_bounds = np.maximum(0, np.floor(image_coords.min(axis=0)).astype(int))
        upper_bounds = np.minimum(shape, np.ceil(image_coords.max(axis=0)).astype(int))
        
        all_relevant_indices = itertools.product(
            *(range(start, stop+1) for start, stop in zip(lower_bounds, upper_bounds))
        )
        num_voxels_in_mask = np.product(1 + upper_bounds - lower_bounds)
        fraction_of_image = num_voxels_in_mask / np.product(shape)
        print(f"Computed mesh bounding box, evaluating {fraction_of_image:.0%} of all image voxels")
        print(f"There are {num_voxels_in_mask} voxels in the bounding box")
    else:

        if not isinstance(mask, np.ndarray):

            mask = nibabel.load(mask).get_fdata()

        if np.isnan(mask).sum() > 0:
            # The mask is defined by the voxels that are not nan
            mask = ~ np.isnan(mask)
        else:
            # Assert the mask is of bool type
            assert np.max(mask) == 1
            assert np.min(mask) == 0


        nonzeros = np.nonzero(mask)
        num_voxels_in_mask = len(nonzeros[0])
        all_relevant_indices = zip(*nonzeros)
        fraction_of_image = num_voxels_in_mask / np.product(output_data.shape)
        print(f"Using mask, evaluating {fraction_of_image:.0%} of all image voxels")
        print(f"There are {num_voxels_in_mask} voxels in the mask")
        if fraction_of_image > 1 - 1e-10 and not allow_full_mask:
            raise ValueError("The supplied mask covers the whole image so you are probably doing something wrong." /
                            " To allow for this behaviour, run with --allow_full_mask")



    # Populate image
    def eval_fenics(f, coords, extrapolation_value):
        try:
            return f(*coords)
        except RuntimeError:
            return extrapolation_value
        
    eps = 1e-12

    try:
        progress = tqdm(total=num_voxels_in_mask)
    except NameError:
        pass

    for xyz_vox in all_relevant_indices:
        xyz_ras = apply_affine(vox2ras, xyz_vox)
        output_data[xyz_vox] = eval_fenics(function, xyz_ras, extrapolation_value)
        try:
            progress.update(1)
        except NameError:
            pass
    
    output_data = np.where(output_data < eps, eps, output_data)
    if mask is not None:
        output_data = np.where(mask, output_data, np.nan)
    # Save output
    output_nii = nibabel.Nifti1Image(output_data, template_image.affine, template_image.header)

    return output_nii, output_data







def make_coordinate_grid(image: np.ndarray, pixelsizes: tuple) -> np.ndarray:
    """
    TODO fix docstring
    Create a (n, d) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)
    """

    if not len(set(image.shape)) == 1:
        assert len(set(pixelsizes)) == 1
        assert pixelsizes[0] == 1
        # Need to change this function if pixelsizes differ
        raise NotImplementedError
    else: 
        n = image.shape[0]

    assert len(pixelsizes) == len(image.shape)

    coordinate_axis = np.linspace(1 / 2, n - 1 / 2, n)

    if len(image.shape) == 2:    
        XX, YY = np.meshgrid(coordinate_axis, coordinate_axis, indexing='ij')
        arr = np.array([XX, YY])

    elif len(image.shape) == 3:
        assert image.shape[-1] > 1
        
        XX, YY, ZZ = np.meshgrid(coordinate_axis, coordinate_axis, coordinate_axis, indexing='ij')
        arr = np.array([XX, YY, ZZ])

    coordinate_grid = np.swapaxes(arr, 0, 1)
    coordinate_grid = np.swapaxes(coordinate_grid, 1, 2)
    
    coordinate_grid[:, 0] *= pixelsizes[0]
    coordinate_grid[:, 1] *= pixelsizes[1]

    if len(image.shape) == 3:
        coordinate_grid = np.swapaxes(coordinate_grid, 2, 3)
        coordinate_grid[:, 2] *= pixelsizes[2]
    
    return coordinate_grid



def init_collocation_points(coords, num_points, t_max, t_min ):

    assert len(coords.shape) == 2, "Assert mask has been applied"

    random_ints = np.random.randint(high=coords.size(0), size=(num_points,))    
    coords = coords[random_ints, :]

    random_times = (np.random.rand(coords.shape[0]))
    # a = lhs(1, coords.shape[0]).flatten().astype(float)

    t = (random_times * (t_max - t_min) + t_min)

    coords[..., -1] = t

    print("Initialized collocation points with mean t = ",
        format(np.mean(t).item(), ".2f"),
        ", min t = ", format(np.min(t).item(), ".2f"),
        ", max t = ", format(np.max(t).item(), ".2f"))

    return coords





def find_bdry(image):
    '''Assuming binary image mask 0 pixels that neighbor 1 pixels'''
    kernels = []
    if image.ndim == 2:
        kernel_shape = (3, 3)
        avoid = 4
    else:
        assert image.ndim == 3

        kernel_shape = (3, 3, 3)
        avoid = 13

    l = np.prod(kernel_shape)
    for i in (set(range(l)) - set((avoid, ))):
        kernel = np.zeros(l, dtype=int)
        kernel[i] = 1
        kernels.append(kernel.reshape(kernel_shape))

    kernels = iter(kernels)
    # The idea here is that an outside boundary pixel should light up
    # at least once when the kernel hits it.
    outside = image == 0

    img = convolve(image, weights=next(kernels), mode='constant', cval=0)
    mask = np.zeros(img.shape, dtype=bool)
    mask[np.where(np.logical_and(img == 1, outside))] = True

    for kernel in kernels:
        img = convolve(image, weights=kernel, mode='constant', cval=0)
        mask_ = np.zeros(img.shape, dtype=bool)
        mask_[np.where(np.logical_and(img == 1, outside))] = True

        mask = np.logical_or(mask, mask_)

    return mask


def find_pepper(image):
    '''Assuming binary image find 1 pixels surrounded by 0'''
    if image.ndim == 2:
        kernel = np.ones((3, 3))
    else:
        assert image.ndim == 3
        kernel = np.ones((3, 3, 3))

    lit = convolve(image, kernel, mode='constant', cval=0)
    # Only the pixel it self lights up
    return np.where(lit == 1)





def find_interior_boundary(image):
    '''Booundary points inside the mask'''
    exterior_boundary = find_bdry(image)
    boundary_boundary = find_bdry(exterior_boundary)
    # Is inside mask and is boundary
    return boundary_boundary * image


def largest_cc(mask):

    # breakpoint()

    mask = np.ascontiguousarray(mask)

    component_masks, ncomps = cc3d.connected_components(
        mask, connectivity=26, return_N=True)
    comp_size = ((i, np.sum(component_masks == i)) for i in range(1, 1+ncomps))
    comp_size = sorted(comp_size, key=lambda p: p[1], reverse=True)

    idx, size = comp_size[0]
    print(f'Largest cc has {size} voxels of {np.prod(mask.shape)}'.format(
        size, np.prod(mask.shape)))
    print(f'Throw out {sum(p[1] for p in comp_size[1:])}')
    largest_cc = np.where(component_masks == idx)

    largest_mask = np.zeros(mask.shape)
    largest_mask[largest_cc] = 1.

    p = find_pepper(largest_mask)
    largest_mask[p] = 0

    # numpy_to_vtk(largest_mask, filename='largest_cc')

    return largest_mask


def get_exterior_boundary(mask):

    largest_mask = largest_cc(mask)

    bdry = find_bdry(largest_mask)

    return bdry



def get_bounding_box_limits(x):
    """ Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)
    for kdim in all_axis:
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = np.diff(mask_i)
        idx_i = np.nonzero(dmask_i)[0]
        if len(idx_i) != 2:
            raise ValueError(
                'Algorithm failed, {} does not have 2 elements!'.format(idx_i))
        bbox.append(slice(idx_i[0] + 1, idx_i[1] + 1))


    return bbox



def cut_to_box(image, mask=None, box_bounds=None):

    if mask is not None:
        box_boundary = get_bounding_box_limits(mask)
    else:
        assert box_bounds is not None
        box_boundary = box_bounds

    xlim_box = [box_boundary[0].start, box_boundary[0].stop]
    ylim_box = [box_boundary[1].start, box_boundary[1].stop]
    
    size = [xlim_box[1] - xlim_box[0], ylim_box[1] - ylim_box[0]]
    

    if len(image.shape) == 3:
        zlim_box = [box_boundary[2].start, box_boundary[2].stop]

        size.append( zlim_box[1] - zlim_box[0])

    # size = [xlim_box[1] - xlim_box[0], ylim_box[1] - ylim_box[0], zlim_box[1] - zlim_box[0]]
    
    size = [np.ceil(x).astype(int) for x in size]

    # returnimage = np.zeros(tuple(size))

    if len(image.shape) == 3:

        returnimage = image[xlim_box[0]:xlim_box[1], ylim_box[0]:ylim_box[1], zlim_box[0]:zlim_box[1],]
    else:
        returnimage = image[xlim_box[0]:xlim_box[1], ylim_box[0]:ylim_box[1]]
        

    # print("cropped shape", returnimage.shape, "box boundary", box_boundary)

    return returnimage
