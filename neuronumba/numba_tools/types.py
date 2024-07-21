import numpy as np
from nptyping import NDArray, Shape

ArrF8_2d = NDArray[Shape["*, *"], np.float64]
ArrF8_1d = NDArray[Shape["*"], np.float64]
