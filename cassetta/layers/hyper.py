"""Utilities to build hyper-networks"""
__all__ = ["HyperNet"]
# builtin
import copy

# externals
import torch
from torch import nn
from torch.utils.data import default_collate
from typing_extensions import Optional, Iterator, List, Union, Set

# internals
from cassetta.core.utils import argalias
from cassetta.backbones.mlp import MLPWrapper, MLP


Module = nn.Module
Tensor = torch.Tensor

_MemoType = Optional[Set[Module]]
_NodesType = Set[Union[str, Module]]


class HyperNet(Module):
    """
    Generic hypernetwork.

    An hyper-network is a network whose weights are generated dynamically
    by a meta-network from a set of input features.

    Its forward pass is: `HyperNet(x, feat) = SubNet(MetaNet(feat))(x)`
    """

    @property
    def inp_channels(self) -> int:
        """Number of channels in the network's input tensor."""
        return self.network.inp_channels

    @property
    def out_channels(self) -> int:
        """Number of channels in the network's output tensor."""
        return self.network.out_channels

    @property
    def inp_features(self) -> int:
        """Number of input features in the hyper-network."""
        return self.hyper.inp_features

    @property
    def out_features(self) -> int:
        """
        Number of output features in the hyper-network.
        Equal to the number of dynamically generated weights in the network.
        """
        return self.hyper.out_features

    @argalias({"inp_features": "inp_channels"})
    def __init__(
        self,
        inp_features: int,
        network: Module,
        backbone: Optional[Module] = None,
        nodes: Optional[List[Union[str, Module]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        inp_features : int
            Number of input meta-features.
        network : Module
            Instantiated sub network, whose weights are dynamically generated.
            This network *will* be modified in place.
        backbone : Module, optional
            Backbone of the hyper-network
        nodes : [sequence of] str or Module, optional
            Names or references of sub-modules, whose weights are
            hyper-generated. Names can include global patterns such as
            '*' or '**'.
            By default, all parameters of the subnetwork are hyper-generated.
        """
        super().__init__()
        if nodes is not None and isinstance(nodes, (str, Module)):
            nodes = [nodes]
        self.nodes = set(nodes) if nodes is not None else None

        # convert parameters to buffers
        self.network = network
        self._preprocess_network_(self.network)

        # make hypernetwork
        nb_weights = sum(w.numel() for w in self._get_weights(network))
        backbone = backbone or MLP(channels=(128,)*6)
        self.hyper = MLPWrapper(inp_features, nb_weights, backbone)

    def _make_chunks(self, x: Tensor) -> Iterator[Tensor]:
        """Cut output of hypernetwork into weights with correct shape"""
        offset = 0
        all_shapes = [p.shape for p in self._get_weights(self.network)]
        for shape in all_shapes:
            numel = torch.Size(shape).numel()
            w = x[offset:offset+numel].reshape(shape)
            offset += numel
            yield w

    def _preprocess_network_(
        self,
        x: Module,
        memo: _MemoType = None,
        nodes: Optional[_NodesType] = None,
        prefix: str = '',
    ) -> None:
        """Convert all "generated" parameters into buffers.

        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        nodes : set[str or Module]
            Set of nodes, whose parameters can be mutated.
        prefix : str
            Full name of the current module.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return
        if nodes is None:
            if self.nodes is None:
                nodes = None
            else:
                nodes = set(self.nodes)

        if nodes is None or x in nodes or self._prefix_in_nodes(prefix, nodes):
            if nodes is not None:
                nodes = nodes.union(set(x.children()))
            param_names = [p[0] for p in x.named_parameters(recurse=False)]
            for name in param_names:
                old = getattr(x, name)
                delattr(x, name)
                x.register_buffer(name, torch.Tensor(old.detach()))
                if not hasattr(x, '_generated_parameters'):
                    setattr(x, '_generated_parameters', [])
                x._generated_parameters.append(name)

        memo.add(x)
        for name, module in x.named_children():
            subprefix = prefix + ('.' if prefix else '') + name
            self._preprocess_network_(module, memo, nodes, subprefix)

    def _postprocess_network_(self, x: Module, memo: _MemoType = None) -> None:
        """Convert all "generated" buffers into parameters

        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, '_generated_parameters', []):
            old = getattr(x, name)
            delattr(x, name)
            setattr(x, name, torch.nn.Parameter(old.detach()))
        if hasattr(x, '_generated_parameters'):
            delattr(x, '_generated_parameters')

        memo.add(x)
        for name, module in x.named_children():
            self._postprocess_network_(module, memo)

    def _prefix_in_nodes(self, prefix: str, nodes: _NodesType) -> bool:
        """Check if a module full-name is in the list of mutable nodes"""
        def _isequal(x, y):
            if x == y:
                return True
            elif y == '*':
                return True
            return False

        for node in nodes:
            if not isinstance(node, str):
                continue
            prefix = prefix.split('.')
            node = node.split('.')
            if '**' in node:
                node = (node[:node.index('**')]
                        + ['*'] * max(0, len(prefix)-len(nodes))
                        + node[node.index('**')+1:])
            if '**' in node:
                raise ValueError(
                    'There can be only one ** ellipsis in pattern'
                )
            if len(node) != len(prefix):
                continue
            if all(_isequal(x, y) for x, y in zip(prefix, node)):
                return True
        return False

    def _get_weights(self, x: Module, memo: _MemoType = None
                     ) -> Iterator[Tensor]:
        """Get all hyper-generated weights of the main module

        This method assumes that `_preprocess_network_` has been called before.

        Parameters
        ----------
        x : Module
            Module to explore.
        memo : set[Module]
            Sub-modules that have already been visited.

        Yields
        ------
        Tensor
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, '_generated_parameters', []):
            yield getattr(x, name)

        memo.add(x)
        for name, module in x.named_children():
            for param in self._get_weights(module, memo):
                yield param

    def _set_weights(
        self,
        x: Module,
        w: Iterator[Tensor],
        memo: _MemoType = None,
    ) -> None:
        """Sets the hyper-generated weights of the main module
        Parameters
        ----------
        x : Module
            Module to explore.
        w : iterator[tensor]
            Iterator over new weights.
        memo : set[Module]
            Sub-modules that have already been visited.
        """
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name in getattr(x, '_generated_parameters', []):
            setattr(x, name, next(w))

        memo.add(x)
        for name, module in x.named_children():
            self._set_weights(module, w, memo)

    def _detach_buffers_(self, x: Module, memo: _MemoType = None) -> None:
        """Detach all buffers in a network"""
        if memo is None:
            memo = set()
        if x in memo:
            return

        for name, buffer in x.named_buffers(recurse=False):
            setattr(x, name, getattr(x, name).detach())

        memo.add(x)
        for name, module in x.named_children():
            self._detach_buffers_(module, memo)

    def make_networks(self, feat: int) -> Iterator[Module]:
        """Generate networks from features

        Parameters
        ----------
        feat : (batch, in_features) tensor
            Input features

        Yields
        ------
        networks : list[Module]
        """
        self._detach_buffers_(self.network)
        weights = self.hyper(feat)
        for batch_weights in weights:
            network = copy.deepcopy(self.network)
            self._set_weights(network, self._make_chunks(batch_weights))
            self._postprocess_network_(network)
            yield network

    def forward(self, feat: Tensor, *args, **kwargs) -> Tensor:
        """
        Parameters
        ----------
        feat : (batch, in_features) tensor
            Input to the hyper-network
        args, kwargs
            Input to the main network.
            Each tensor must have a batch dimension.

        Returns
        -------
        out : (batch, ...) tensor
            Output of the main network.
            Can be a collated tensor a collated collection of tensors.
        """
        nbatch = len(feat)

        # generate hyper weights
        weights = self.hyper(feat)

        if len(weights) == 1:
            self._set_weights(self.network, self._make_chunks(weights[0]))
            return self.network(*args, **kwargs)

        # uncollate arguments
        args = [
            [
                x[i:i+1] if torch.is_tensor(x) else x
                for x in args
            ]
            for i in range(nbatch)
        ]
        kwargs = [
            {
                k: (x[i:i+1] if torch.is_tensor(x) else x)
                for k, x in kwargs.items()
            }
            for i in range(nbatch)
        ]

        # we have to loop over batches because network weights cannot have
        # a batch dimension
        output = []
        for w, a, k in zip(weights, args, kwargs):
            self._set_weights(self.network, self._make_chunks(w))
            output.append(self.network(*a, **k))
        return default_collate(output)
