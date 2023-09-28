import torch as torch
import numpy as np
from torchvision import transforms


# Orginal dataset class for 2D images
class BinomDataset(torch.utils.data.Dataset):
    """
    Returns a BinomDataset that will randomly split an image into input and target using a binomial distribution for each pixel.
        Parameters:
            data (numpy array): a 3D numpy array (z, y, x) with integer photon counts
            windowSize (int): size of XY window to be used in random crops (default is 64)
            minPSNR (float): minimum pseudo PSNR of sampled data (see supplement)
            maxPSNR (float): maximum pseudo PSNR of sampled data (see supplement)
            virtSize (int): virtual size of dataset (default is None, i.e., the real size)
            augment (bool): use 8-fold data augmentation (default is False)
            maxProb (float): the maximum success probability for binomial splitting
        Returns:
            dataset
    """

    def __init__(
        self,
        data: np.ndarray = None,
        windowSize: int = 256,
        minPSNR: float = -40.0,
        maxPSNR: float = 40.0,
        virtSize: int = None,
        augment: bool = True,
        maxProb: float = 0.99,
    ):
        self.data: torch.Tensor = torch.from_numpy(data.astype(np.int32))
        self.crop = transforms.RandomCrop(windowSize)
        self.flipH = transforms.RandomHorizontalFlip()
        self.flipV = transforms.RandomVerticalFlip()
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.windowSize = windowSize
        self.maxProb = maxProb
        self.std = data.std()
        self.virtSize = virtSize
        self.augment = augment

    # Return either the real size or the virtual size of the dataset
    def __len__(self) -> int:
        if self.virtSize is not None:
            return self.virtSize
        else:
            return self.data.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        idx_ = idx
        if self.virtSize is not None:
            idx_ = np.random.randint(self.data.shape[0])
        img = self.crop(self.data[idx_])
        uniform = np.random.rand() * (self.maxPSNR - self.minPSNR) + self.minPSNR

        level = (10 ** (uniform / 10.0)) / (img.type(torch.float).mean().item() + 1e-5)
        level = min(level, self.maxProb)

        binom = torch.distributions.binomial.Binomial(
            total_count=img, probs=torch.tensor([level])
        )
        imgNoise = binom.sample()

        img = (img - imgNoise)[None, ...].type(torch.float)
        img = img / (img.mean() + 1e-8)

        imgNoise = imgNoise[None, ...].type(torch.float)
        out = torch.cat((img, imgNoise), dim=0)

        if not self.augment:
            return out
        else:
            if np.random.rand() < 0.5:
                out = torch.transpose(out, -1, -2)
            return self.flipV(self.flipH(out))


import torch
import torch.utils.data
import numpy as np
from torchvision import transforms


class BinomDataset3D(torch.utils.data.Dataset):
    """
    Returns a BinomDataset that will randomly split an image into input and target using a binomial distribution for each pixel.
        Parameters:
        data (numpy array): a 3D numpy array (z, y, x) with integer photon counts
            unitThickness (int): thickness of the unit volume
            windowSize (int | tuple): size of XY window to be used in crops (default is (16, 64, 64))
            minPSNR (float): minimum pseudo PSNR of sampled data (see supplement)
            maxPSNR (float): maximum pseudo PSNR of sampled data (see supplement)
            augment (bool): use 8-fold data augmentation (default is False)
            maxProb (float): the maximum success probability for binomial splitting
        Returns:
            dataset
    """

    def __init__(
        self,
        data: np.ndarray = None,
        unitThickness: int = 16,
        windowSize: int | tuple = (16, 64, 64),  # (z, x, y)
        rand_crop: bool = True,
        top_corner: tuple = (0, 0, 0),
        minPSNR: float = -40.0,
        maxPSNR: float = 40.0,
        augment: bool = True,
        maxProb: float = 0.99,
    ):
        super().__init__()
        self.img = data
        # define crop window size
        if isinstance(windowSize, int):
            self.windowSize = (windowSize, windowSize, windowSize)
        else:
            self.windowSize = windowSize
        assert len(windowSize) == 3, "windowSize must be a tuple of length 3"
        assert windowSize[0] <= data.shape[-3], "windowSize_z must <= data_z"
        assert windowSize[1] <= data.shape[-2], "windowSize_x must <= data_x"
        assert windowSize[2] <= data.shape[-1], "windowSize_y must <= data_y"
        assert windowSize[0] <= unitThickness, "windowSize[0] must be <= unitThickness"
        # convert data to torch tensor
        if np.max(data) > 255:
            self.data = torch.from_numpy(data.astype(np.int32))
        else:
            self.data = torch.from_numpy(data.astype(np.uint8))
        self.rand_crop = rand_crop
        self.top_corner = top_corner
        
        # Random arguments
        self.flipH = transforms.RandomHorizontalFlip()
        self.flipV = transforms.RandomVerticalFlip()

        # save parameters
        self.unitThickness = unitThickness
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.maxProb = maxProb
        self.dataLength = self.data.shape[-3] - unitThickness
        self.std = data.std()
        self.augment = augment

    # Return the length of the dataset
    def __len__(self) -> int:
        return self.dataLength

    # Return a cropped image from the dataset
    def __getitem__(self, idx: int) -> torch.Tensor:
        idx_ = idx
        assert idx_ < self.dataLength, "idx must < dataLength"
        img = self._crop_3d(idx=idx_)
        # grenerate random noise from binomial distribution
        uniform = np.random.rand() * (self.maxPSNR - self.minPSNR) + self.minPSNR
        level = (10 ** (uniform / 10.0)) / (img.type(torch.float).mean().item() + 1e-5)
        level = min(level, self.maxProb)
        binom = torch.distributions.binomial.Binomial(
            total_count=img, probs=torch.tensor([level])
        )
        imgNoise = binom.sample()
        # normalize the data
        img = (img - imgNoise)[None, ...].type(torch.float)
        img = img / (img.mean() + 1e-8)

        imgNoise = imgNoise[None, ...].type(torch.float)
        out = torch.cat((img, imgNoise), dim=0)
        if not self.augment:
            return out
        else:
            if np.random.rand() < 0.5:
                out = torch.transpose(out, -1, -2)
            return self.flipV(self.flipH(out))

    def _crop_3d(self, idx: int) -> tuple:
        """
        Return a tuple of indexes for cropping the image.

            Parameters:
                idx (int): index of the image to be cropped
                top_corner (tuple): top corner of the crop (z, x, y)
                rand_crop (bool): whether to randomly crop the image or not
            Returns:
                tuple of indexes for cropping the image
        """
        z = self.unitThickness
        x = self.img.shape[-2]
        y = self.img.shape[-1]
        z_crop, x_crop, y_crop = self.windowSize  # size of the crop
        # define the starting point of crop
        if self.rand_crop:
            z_start = np.random.randint(idx, idx + z - z_crop + 1)
            x_start = np.random.randint(0, x - x_crop + 1)
            y_start = np.random.randint(0, y - y_crop + 1)
        else:
            z_start, x_start, y_start = self.top_corner
            z_start += idx
        # define the end point of the crop
        z_end = z_start + z_crop
        x_end = x_start + x_crop
        y_end = y_start + y_crop
        # return the data
        return self.data[z_start:z_end, x_start:x_end, y_start:y_end]