from torch.optim import Adam
from torch.nn import Linear
from cassetta.io import make_loadable
from cassetta.io.loadable import LoadableMixin, load
from cassetta.io.optim import LoadableOptimizerMixin


class _Loadable(LoadableMixin):
    def __init__(self, a=1, b=2):
        super().__init__()
        self.a = a
        self.b = b


def test_loadable_mixin() -> None:
    reference_object = _Loadable(3, b=4)
    reference_state = reference_object.serialize()

    # load using class loader
    loaded_object = LoadableMixin.load(reference_state)
    loaded_state = loaded_object.serialize()

    assert (
        reference_object.a == loaded_object.a and
        reference_object.b == loaded_object.b
    ), "Reference and loaded objects differ."

    assert reference_state == loaded_state, \
        "Reference and loaded states differ."

    # test load function (should be calling LoadableMixin.load)
    loaded_object = load(reference_state)
    loaded_state = loaded_object.serialize()

    assert (
        reference_object.a == loaded_object.a and
        reference_object.b == loaded_object.b
    ), "Reference and loaded objects differ."

    assert reference_state == loaded_state, \
        "Reference and loaded states differ."


def test_loadable_local() -> None:

    class _LocalLoadable(LoadableMixin):
        def __init__(self, a=1, b=2):
            super().__init__()
            self.a = a
            self.b = b

    reference_object = _LocalLoadable(3, b=4)
    reference_state = reference_object.serialize()
    loaded_object = _LocalLoadable.load(reference_state)
    loaded_state = loaded_object.serialize()

    assert (
        reference_object.a == loaded_object.a and
        reference_object.b == loaded_object.b
    ), "Reference and loaded objects differ."

    assert reference_state == loaded_state, \
        "Reference and loaded states differ."


def test_loadable_optim() -> None:
    module = Linear(1, 1)
    optim = make_loadable(Adam)(module.parameters())
    state = optim.serialize()
    loaded_object = LoadableOptimizerMixin.load(state, module.parameters())
    loaded_state = loaded_object.serialize()

    assert state == loaded_state, \
        "Reference and loaded states differ."
