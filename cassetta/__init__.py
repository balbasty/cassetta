"""
This package is a versatile deep-learning toolbox for PyTorch,
tailored to researchers working with N-dimensional vision problems,
and more specifically medial imaging problems.

It is intended to provide building blocks for a wide variety of
architectures, as well as a set of pre-defined backbones, as well as
a few task-specific models (segmentation, registration, synthesis, ...).

It will not provide domain-specific tools with dedicated pre- and post-
processing pipelines. However, such high-level tools can be implemented
using this toolbox.

Modules
-------
models
    Task-specific models.
backbones
    Task-agnostic architectures to use as backbones in models.
layers
    Building blocks for backbones and models.
losses
    Differentiable functions to optimize during training.
metrics
    Non-differentiable functions to compute during validation.
training
    Tools to train networks.
inference
    Tools to apply networks to unseed data.
functional
    Lower-level functional utilities.
io
    Input/output.
core
    Core utilities, mostly intended for internal use.
"""

from . import core              # noqa: F401
from . import functional        # noqa: F401
from . import inference         # noqa: F401
from . import training          # noqa: F401
from . import metrics           # noqa: F401
from . import losses            # noqa: F401
from . import layers            # noqa: F401
from . import backbones         # noqa: F401
from . import models            # noqa: F401
from . import io                # noqa: F401

from . import _version
__version__ = _version.get_versions()['version']
