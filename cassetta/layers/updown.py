from torch import nn


class Upsample(nn.Module):
    """Upsample a tensor using corners as anchors"""

    def __init__(self, factor=2, anchor='center'):
        """

        Parameters
        ----------
        factor : int, Upsampling factor
        anchor : {'center', 'edge'}
            Use the center or the edges of the corner voxels as anchors

        """
        super().__init__()
        self.factor = factor
        self.anchor = anchor

    def forward(self, image, shape=None):
        """

        Parameters
        ----------
        image : (B, D, *shape) tensor
        shape : list[int], optional

        Returns
        -------
        image : (B, D, *shape) tensor

        """
        factor = None if shape else self.factor
        return warps.upsample(image, factor, shape, self.anchor)


class Downsample(nn.Module):
    """Downsample a tensor using corners as anchors"""

    def __init__(self, factor=2, anchor='center'):
        """

        Parameters
        ----------
        factor : int, Downsampling factor
        anchor : {'center', 'edge'}
            Use the center or the edges of the corner voxels as anchors

        """
        super().__init__()
        self.factor = factor
        self.anchor = anchor

    def forward(self, image, shape=None):
        """

        Parameters
        ----------
        image : (B, D, *shape) tensor
        shape : list[int], optional

        Returns
        -------
        image : (B, D, *shape) tensor

        """
        factor = None if shape else self.factor
        return warps.downsample(image, factor, shape, self.anchor)


class UpsampleConvLike(nn.Module):
    """Upsample an image the same way a transposed convolution would"""

    def __init__(self, kernel_size, stride=2, padding=0):
        """

        Parameters
        ----------
        kernel_size : [list of] int
        stride : [list of] int
        padding : [list of] int

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, flow, shape=None):
        """

        Parameters
        ----------
        flow : (B, D, *shape) tensor
        shape : list[int], optional

        Returns
        -------
        flow : (B, D, *shape) tensor

        """
        return warps.upsample_convlike(
            flow,  self.kernel_size, self.stride, self.padding, shape)


class DownsampleConvLike(nn.Module):
    """Downsample an image the same way a strided convolution would"""

    def __init__(self, kernel_size, stride=2, padding=0):
        """

        Parameters
        ----------
        kernel_size : [list of] int
        stride : [list of] int
        padding : [list of] int

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, flow):
        """

        Parameters
        ----------
        flow : (B, D, *shape) tensor

        Returns
        -------
        flow : (B, D, *shape) tensor

        """
        return warps.downsample_convlike(
            flow, self.kernel_size, self.stride, self.padding)
