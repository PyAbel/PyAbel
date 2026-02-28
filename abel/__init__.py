import warnings
warnings.warn(
    "The 'abel' package has been renamed to 'pyabel'. "
    "Please update your imports: replace 'import abel' with 'import pyabel'. "
    "The 'abel' compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from pyabel import *  # noqa: F401, F403
from pyabel import (  # noqa: F401
    _deprecated, _deprecate,
    basex, benchmark, dasch, daun, direct,
    hansenlaw, linbasex, onion_bordas, rbasex,
    tools, transform, Transform, center_image,
    __version__,
)
