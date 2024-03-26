__all__ = [
    'import_qualname',
    'import_fullname',
]
from importlib import import_module


def import_qualname(module, qualname):
    """
    Import or get object from its qualified name

    Parameters
    ----------
    module : module or str or dict
        Instantiated module, or module path such as `"package.module"`,
        or a dictionary of objects such as `locals()`.
    qualname : str
        Qualified name, such as `"MyClass.SubClass.attribute"`.

    Returns
    -------
    object
    """
    qualname = list(qualname.split('.'))

    obj = module
    if isinstance(obj, str):
        obj = import_module(obj)

    while qualname:
        if isinstance(obj, dict):
            obj = obj[qualname.pop(0)]
        else:
            obj = getattr(obj, qualname.pop(0))

    return obj


def import_fullname(fullname):
    """
    Import object from its fully qualified name, where we don't know
    which part is the module path, and which part if the qualified path.

    Parameters
    ----------
    fullname : str
        Fully qualified name, such as `"package.module.MyClass.attribute"`.

    Returns
    -------
    object
    """
    fullname = list(fullname.split('.'))

    # import module and submodules
    modulename = fullname.pop(0)
    module = import_module(modulename)
    while fullname:
        if hasattr(module, fullname[0]):
            break
        try:
            module = import_module(modulename + '.' + fullname[0])
            modulename += '.' + fullname.pop(0)
        except (ImportError, ModuleNotFoundError):
            break
    return import_qualname(module, '.'.join(fullname))
