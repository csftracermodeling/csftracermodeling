import numpy as np



from abc import ABC, abstractmethod
from tracerdiffusion.utils import make_coordinate_grid

class Domain(ABC):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def sample(self):
        pass



class ImageDomain(Domain):

    def __init__(self, mask: np.ndarray, pixelsizes: tuple, verbosity: int=1, *args, **kwargs) -> None:
        super(ImageDomain, self).__init__(*args, **kwargs)

        self.mask = mask

        assert len(set(pixelsizes)) == 1
        assert pixelsizes[0] == 1

        self.dx = pixelsizes[0] / 2

        self.pixelsizes = pixelsizes

        self.d = len(mask.shape)

        coordinate_grid = make_coordinate_grid(image=self.mask, pixelsizes=self.pixelsizes)

        self.voxel_center_coordinates = coordinate_grid[self.mask]

        if verbosity == 1:
            print("Initialized coordinates ")

        self.min = np.min(self.voxel_center_coordinates, axis=0)
        self.max = np.max(self.voxel_center_coordinates, axis=0)

    def bounds(self) -> tuple:

        return self.min, self.max
    
    def sample(self, n: int) -> np.ndarray:

        random_ints = np.random.randint(low=0, high=self.voxel_center_coordinates.shape[0], size=(n,))    
        random_floats = np.random.rand(n, self.voxel_center_coordinates.shape[-1])
        
        random_floats -= self.dx

        random_voxel_coordinates = self.voxel_center_coordinates[random_ints, :] + random_floats

        return random_voxel_coordinates
