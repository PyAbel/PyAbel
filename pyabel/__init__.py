from ._version import __version__

'''
Tools to warn about deprecated functions and arguments. Usage:

from abel import _deprecated, _deprecate

def func_old(...):
    """Deprecated function. Use :func:`func_new` instead."""
    _deprecate('abel.submodule.func_old() '
               'is deprecated, use abel.module.func_new() instead.')
    ...

def func(..., arg_old=_deprecated, arg_new):
    if arg_old is not _deprecated:
        _deprecate('abel.submodule.func() '
                   'argument "arg_old" is deprecated, use "arg_new" instead.')
        arg_new = arg_old
    ...

In unit tests:

# to suppress deprecation warnings
from warnings import catch_warnings, simplefilter

def test_something():
    ...
    with catch_warnings():
        simplefilter('ignore', category=DeprecationWarning)
        # test deprecated function/arguments
        ...
    ...

'''
from warnings import warn, filterwarnings
# class for documentation format
class __deprecated:
    def __repr__(self):
        return '<deprecated>'
# marker of deprecated parameters
_deprecated = __deprecated()
# print deprecation warning
def _deprecate(msg):
    warn(msg, DeprecationWarning, stacklevel=3)
    # 3rd level is from where the deprecated abel function is called:
    # level 0   level 1         level 2      level 3
    # warn() <- _deprecate() <- abel...() <- user code
# enable deprecation warnings (ignored by default) for abel
filterwarnings('default', r'^abel\.', category=DeprecationWarning)

from . import basex
from . import benchmark
from . import dasch
from . import daun
from . import direct
from . import hansenlaw
from . import linbasex
from . import onion_bordas
from . import rbasex
from . import tools
from . import transform
from .transform import Transform
from .tools.center import center_image
