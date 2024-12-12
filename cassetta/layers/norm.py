r"""
# Normalisation layers

In general, normalisation consists of computing the Z-score of a
tensor along one or more dimensions, followed by a learnable
affine transformation (along the same or a different set of dimensions).
In other tems:

$$
    y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \varepsilon}}
    \dot \gamma + \beta
$$

What separates different types of normalisation is the dimension(s)
along which they operate:

* [`BatchNorm`][cassetta.layers.norm.BatchNorm] computes statistics
  and learns affine parameters _per channel_
  (and therefore across batch and spatial dimensions);
* [`InstanceNorm`][cassetta.layers.norm.InstanceNorm] computes statistics
  _per channel and per batch_ (and therefore across spatial dimensions),
  and learns affine parameters per channel;
* [`LayerNorm`][cassetta.layers.norm.LayerNorm] computes statistics
  _per batch_ (and therefore across channels and spatial dimensions),
  and learns a single set of affine parameters.

"""
__all__ = [
    'make_norm',
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
]
# externals
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormBase
from torch.nn.modules.instancenorm import _InstanceNorm as InstanceNormBase
from typing_extensions import Optional, Union, List

# internals
from cassetta.core.typing import DeviceType, DataType, NormType
from cassetta.core.utils import to_torch_dtype, ensure_list


def make_norm(
    norm: NormType,
    channels: Union[int, List[int]],
    affine: bool = True,
    **kwargs
):
    """
    Instantiate a normalization module.

    To be accepted in a `nn.Sequential` module or in a `nn.ModuleList`,
    a norm **must** be a `nn.Module`. This function takes other
    forms of "norm parameters" that are typically passed to the
    constructor of larger models, and generate the corresponding
    instantiated Module.

    A norm-like value can be a `nn.Module` subclass, which is
    then instantiated, or a callable function that returns an
    instantiated Module. It can also be the name of a none normalization:
    `"batch"`, `"layer"`, or `"instance"`.

    It is useful to accept all these cases as they allow to either:

    * have a learnable norm specific to this module
    * have a learnable norm shared with other modules
    * have a non-learnable norm

    Parameters
    ----------
    norm : NormType
        An already instantiated `nn.Module`, or a `nn.Module` subclass,
        or a callable that retgurns an instantiated `nn.Module`, or the
        name of a known normalization: `"batch"` `"layer"`, or `"instance"`.
    channels : [list of] int
        Number of channels
    affine : bool
        Include a learnable affine transform.
    kwargs : dict
        Additional parameters to pass to the constructor or function.

    Returns
    -------
    norm : Module
        An instantiated `nn.Module`.
    """
    kwargs['affine'] = affine

    if not norm:
        return None

    if isinstance(norm, nn.Module):
        return norm

    if norm is True:
        norm = 'batch'

    if isinstance(norm, int):
        return nn.GroupNorm(norm, channels, **kwargs)

    if isinstance(norm, str):
        norm = norm.lower()
        if 'instance' in norm:
            norm = InstanceNorm
        elif 'layer' in norm:
            norm = LayerNorm
        elif 'batch' in norm:
            norm = BatchNorm
        else:
            raise ValueError(f'Unknown normalization "{norm}"')

    norm = norm(channels, **kwargs)

    if not isinstance(norm, nn.Module):
        raise ValueError('Normalization did not instantiate a Module')
    return norm


class BatchNorm(BatchNormBase):
    r"""
    Batch normalisation.

    Computes statistics and learns affine parameters _per channel_.
    """

    # There's really nothing dimension-specific in BatchNorm.
    # The 1d/2d/3d specialization only check dimensions.

    def __init__(
        self,
        channels: Union[int, List[int]],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dim: Union[int, List[int]] = 1,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Parameters
        ----------
        channels : [list of] int
            Number of input channels.
        eps : float
            Value added to the denominator for numerical stability.
        momentum : float
            The value used for the `running_mean` and `running_var`
            computation. Can be set to `None` for cumulative moving average
            (i.e. simple average).
        affine : bool
            Use learnable affine parameters.
        track_running_stats : bool
            Track the running mean and variance. If `False`,
            this module does not track such statistics, and initializes
            statistics buffers as `None`. When these buffers are `None`,
            this module always uses batch statistics in both training
            and eval modes.
        dim : [list of] int
            Indices of channel dimensions.
            Statistics are computed per channel.
        """
        dtype = to_torch_dtype(dtype)
        channels = torch.Size(ensure_list(channels)).numel()
        super().__init__(
            channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype,
            device=device,
        )
        self.dim = ensure_list(dim)

    def _check_input_dim(self, input: Tensor) -> None:
        pass

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (*batch, *channel, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (*batch, *channel, *spatial) tensor
            Output tensor
        """
        dim = [d if d >= 0 else d + inp.ndim for d in self.dim]
        if dim == [1]:
            return super().forward(inp)

        # move channel dimensions to the left
        inp = inp.movedim(dim, list(range(len(self.dim))))
        # flatten as (channels x batch) matrix
        shape = inp.shape
        inp = inp.reshape([shape[:len(dim)].numel(), -1])
        # batch norm
        out = super().forward(inp.T).T
        # unflatten
        out = out.reshape(shape)
        # move back channels to native order
        out = out.movedim(list(range(len(dim))), dim)
        return out


class InstanceNorm(InstanceNormBase):
    r"""
    Instance normalisation.

    Computes statistics _per batch per channel_ and learns affine
    parameters _per channel_.

    Instance normalisation is equivalent to group normalisation where the
    number of groups is one.
    """

    # There's really nothing dimension-specific in InstanceNorm.
    # The 1d/2d/3d specialization only check dimensions.

    def __init__(
        self,
        channels: Union[int, List[int]],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        dim: Union[int, List[int]] = 1,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Parameters
        ----------
        channels : [list of] int
            Number of input channels.
        eps : float
            Value added to the denominator for numerical stability.
        momentum : float
            The value used for the `running_mean` and `running_var`
            computation. Can be set to `None` for cumulative moving average
            (i.e. simple average).
        affine : bool
            Use learnable affine parameters.
        track_running_stats : bool
            Track the running mean and variance. If `False`,
            this module does not track such statistics, and initializes
            statistics buffers as `None`. When these buffers are `None`,
            this module always uses batch statistics in both training
            and eval modes.
        dim : [list of] int
            Indices of channel dimensions.
            Statistics are computed per batch per channel.
            All dimensions left to the first channel dimension are
            considered batch dimensions.
        """
        dtype = to_torch_dtype(dtype)
        channels = torch.Size(ensure_list(channels)).numel()
        super().__init__(
            channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype,
            device=device,
        )
        self.dim = ensure_list(dim)

    def _check_input_dim(self, input):
        pass

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, channels, *size) tensor
            Input tensor

        Returns
        -------
        out : (B, channels, *size) tensor
            Output tensor
        """
        dim = [d if d >= 0 else d + inp.ndim for d in self.dim]
        if dim == [1]:
            if self.affine and input.size(1) != self.channels:
                raise ValueError("Wrong number of channels")
            return self._apply_instance_norm(inp)

        # guess number of batch dimensions
        nb = min(dim)
        nc = len(dim)
        # move channel dimensions to the left
        inp = inp.movedim(dim, list(range(nc)))
        # flatten as (channels x batch x spatial) matrix
        shape = inp.shape
        inp = inp.reshape([shape[:nc].numel(), shape[nc:nc+nb].nume(), -1])
        # check
        if self.affine and input.size(0) != self.channels:
            raise ValueError("Wrong number of channels")
        # batch norm
        inp = inp.permute([1, 0, 2])
        out = self._apply_instance_norm(inp)
        out = out.permute([1, 0, 2])
        # unflatten
        out = out.reshape(shape)
        # move back channels to native order
        out = out.movedim(list(range(nc)), dim)
        return out


class LayerNorm(nn.GroupNorm):
    r"""
    Layer normalisation.

    Computes statistics _per batch_ and learns a single set of affine
    parameters.

    Layer normalisation is equivalent to group normalisation where the
    number of groups equals the number of channels.
    """

    # LayerNorm = GroupNorm with `num_groups=num_channels`

    def __init__(
        self,
        channels: Union[int, List[int]],
        eps: float = 1e-5,
        affine: bool = True,
        dim: Union[int, List[int]] = 1,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None
    ) -> None:
        """
        Parameters
        ----------
        channels : [list of] int
            Number of input channels.
        eps : float
            Value added to the denominator for numerical stability.
        affine : bool
            Use learnable affine parameters.
        dim : [list of] int
            Indices of channel dimensions.
            Statistics are computed per batch.
            All dimensions left to the first channel dimension are
            considered batch dimensions.
        """
        channels = torch.Size(ensure_list(channels)).numel()
        dtype = to_torch_dtype(dtype)
        super().__init__(
            channels, channels, eps, affine, device, dtype
        )
        self.dim = ensure_list(dim)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (*batch, *channels, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (*batch, *channels, *spatial) tensor
            Output tensor
        """
        dim = [d if d >= 0 else d + inp.ndim for d in self.dim]
        if dim == [1]:
            return super().forward(inp)

        # guess number of batch dimensions
        nb = min(dim)
        nc = len(dim)
        # move channel dimensions to the left
        inp = inp.movedim(dim, list(range(nc)))
        # flatten as (channels x batch x spatial) matrix
        shape = inp.shape
        inp = inp.reshape([shape[:nc].numel(), shape[nc:nc+nb].nume(), -1])
        # check
        if self.affine and input.size(0) != self.channels:
            raise ValueError("Wrong number of channels")
        # batch norm
        inp = inp.permute([1, 0, 2])
        out = super().forward(inp)
        out = out.permute([1, 0, 2])
        # unflatten
        out = out.reshape(shape)
        # move back channels to native order
        out = out.movedim(list(range(nc)), dim)
        return out
