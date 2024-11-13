__all__ = [
    'make_norm',
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
]
# externals
from torch import nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormBase
from torch.nn.modules.instancenorm import _InstanceNorm as InstanceNormBase

# internals
from cassetta.core.typing import DeviceType, DataType, NormType, Optional
from cassetta.core.utils import to_torch_dtype


def make_norm(
    norm: NormType,
    channels: int,
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
    channels : int
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
    # There's really nothing dimension-specific in BatchNorm.
    # The 1d/2d/3d specialization only check dimensions.

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Parameters
        ----------
        nb_chanels : int
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
        """
        dtype = to_torch_dtype(dtype)
        super().__init__(
            channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype,
            device=device,
        )

    def _check_input_dim(self, input):
        pass


class InstanceNorm(InstanceNormBase):
    # There's really nothing dimension-specific in InstanceNorm.
    # The 1d/2d/3d specialization only check dimensions.

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Parameters
        ----------
        nb_chanels : int
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
        """
        dtype = to_torch_dtype(dtype)
        super().__init__(
            channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype,
            device=device,
        )

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
        if self.affine and input.size(1) != self.channels:
            raise ValueError("Wrong number of channels")
        return self._apply_instance_norm(inp)


class LayerNorm(nn.GroupNorm):
    # LayerNorm = GroupNorm with `num_groups=num_channels`

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None
    ) -> None:
        """
        Parameters
        ----------
        nb_chanels : int
            Number of input channels.
        eps : float
            Value added to the denominator for numerical stability.
        affine : bool
            Use learnable affine parameters.
        """
        dtype = to_torch_dtype(dtype)
        super().__init__(
            channels, channels, eps, affine, device, dtype
        )
