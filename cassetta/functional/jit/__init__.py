r"""
+-------------------------------------------------------------------+-------------------------------------------------------+
|[**Indexing**][cassetta.functional.jit.indexing]                                                                           |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`ind2sub`][cassetta.functional.jit.ind2sub]                      | Convert linear indices into sub indices (i, j, k).    |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`sub2ind`][cassetta.functional.jit.sub2ind],                     | Convert sub indices (i, j, k) into linear indices.    |
| [`sub2ind_list`][cassetta.functional.jit.sub2ind_list]            |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [**Math**][cassetta.functional.jit.math]                                                                                  |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`square`][cassetta.functional.jit.square],                       | `x**2`                                                |
| [`square_`][cassetta.functional.jit.square_]                      |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`cube`][cassetta.functional.jit.cube],                           | `x**3`                                                |
| [`cube_`][cassetta.functional.jit.cube_]                          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`pow4`][cassetta.functional.jit.pow4],                           | `x**4`                                                |
| [`pow4_`][cassetta.functional.jit.pow4_]                          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`pow5`][cassetta.functional.jit.pow5],                           | `x**5`                                                |
| [`pow5_`][cassetta.functional.jit.pow5_]                          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`pow6`][cassetta.functional.jit.pow6],                           | `x**6`                                                |
| [`pow6_`][cassetta.functional.jit.pow6_]                          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`pow7`][cassetta.functional.jit.pow7],                           | `x**7`                                                |
| [`pow7_`][cassetta.functional.jit.pow7_]                          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`floor_div`][cassetta.functional.jit.floor_div],                 | `floor(x / y)`                                        |
| [`floor_div_int`][cassetta.functional.jit.floor_div_int]          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`trunc_div`][cassetta.functional.jit.trunc_div],                 | `trunc(x / y)`                                        |
| [`trunc_div_int`][cassetta.functional.jit.trunc_div_int]          |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [**Meshgrid**][cassetta.functional.jit.meshgrid]                                                                          |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`meshgrid_list_ij`][cassetta.functional.jit.meshgrid_list_ij]    | Meshgrid with `indexing="ij"`                         |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`meshgrid_list_xy`][cassetta.functional.jit.meshgrid_list_xy]    | Meshgrid with `indexing="xy"`                         |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [**Python objects**][cassetta.functional.jit.python]                                                                      |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`pad_list_int`][cassetta.functional.jit.pad_list_int]            | Pad a list                                            |
| [`pad_list_float`][cassetta.functional.jit.pad_list_float]        |                                                       |
| [`pad_list_str`][cassetta.functional.jit.pad_list_str]            |                                                       |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`any_list_bool`][cassetta.functional.jit.any_list_bool]          | `any(list)`                                           |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`all_list_bool`][cassetta.functional.jit.all_list_bool]          | `all(list)`                                           |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`prod_list_int`][cassetta.functional.jit.prod_list_int]          | `prod(list)`                                          |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`sum_list_int`][cassetta.functional.jit.sum_list_int]            | `sum(list)`                                           |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`reverse_list_int`][cassetta.functional.jit.reverse_list_int]    | `reversed(list)`                                      |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`cumprod_list_int`][cassetta.functional.jit.cumprod_list_int]    | Cumulative product                                    |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [**Tensors**][cassetta.functional.jit.tensors]                                                                            |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`prod_list_tensor`][cassetta.functional.jit.prod_list_tensor]    | `prod(list[tensor])`                                  |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`sum_list_tensor`][cassetta.functional.jit.sum_list_tensor]      | `sum(list[tensor])`                                   |
+-------------------------------------------------------------------+-------------------------------------------------------+
| [`movedim`][cassetta.functional.jit.movedim]                      | `movedim(tensor, src, dst)`                           |
+-------------------------------------------------------------------+-------------------------------------------------------+
"""  # noqa: E501
__all__ = []

from cassetta.core.utils import import_submodules

import_submodules([
    'indexing',         # indexing utilities
    'math',             # math helpers (backward compatibility)
    'meshgrid',         # backward compatible meshgrid
    'python',           # torchscript functions for builtins
    'tensors',          # torchscript functions for tensors
], __name__, __all__, True)
