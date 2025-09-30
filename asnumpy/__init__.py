from .lib import * 
from .lib import init, set_device # 哄pylance的 其实可以不写
from .lib import __all__ as __lib_all__
from .io import save, savez, savez_compressed, load

__all__ = __lib_all__ + ['save', 'savez', 'savez_compressed', 'load']

init()
set_device(0)
