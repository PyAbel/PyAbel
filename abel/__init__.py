from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ._version import __version__

# Temporary tools to warn about deprecated functions and parameters.
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
# enable deprecation warnings (ignored by default since Python 2.7) for abel
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
