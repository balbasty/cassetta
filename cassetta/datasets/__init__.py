"""
## Overview

Datasets organize and allow for dynamic preprocessing of data.

Modules
-------
supervised
    Datasets for supervised learning tasks with paired data (input-output)
"""
from cassetta.core.utils import import_submodules

import_submodules([
    'supervised',
], __name__, True)
