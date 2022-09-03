from .imgarray import ImgArray
from .labeledarray import LabeledArray
from .label import Label
from .phasearray import PhaseArray
from .specials import PropArray
from .lazy import LazyImgArray

# install deprecations

try:
    from ._deprecated import Funcs

    for k, v in Funcs.items():
        setattr(ImgArray, k, v)

except Exception:
    pass
