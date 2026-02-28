import sys
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

# Register all pyabel submodules under abel.* so that
# `import abel.basex`, `from abel.tools.vmi import ...`, etc. keep working.
for _mod in [
    'basex', 'benchmark', 'dasch', 'daun', 'direct',
    'hansenlaw', 'linbasex', 'onion_bordas', 'rbasex',
    'tools', 'transform',
]:
    sys.modules[f'abel.{_mod}'] = sys.modules[f'pyabel.{_mod}']

for _mod in [
    'analytical', 'center', 'circularize', 'io', 'math',
    'polar', 'polynomial', 'symmetry', 'transform_pairs', 'vmi',
]:
    sys.modules[f'abel.tools.{_mod}'] = sys.modules[f'pyabel.tools.{_mod}']

# Also register the optional Cython extension if it was built.
if 'pyabel.lib.direct' in sys.modules:
    sys.modules['abel.lib.direct'] = sys.modules['pyabel.lib.direct']
