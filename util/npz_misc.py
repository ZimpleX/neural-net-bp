import numpy as np
from logf.printf import printf


def lossy_normalize(path_in, path_out):
    """
    Normalize each [-1,1] floating point image to [0,255] utf-8,
    then normalize back to [-1,1].
    This is intentional to see how the precision of data types 
    affects the classification results.
    """
    f_in = np.load(path_in)
    d = f_in['data']
    assert d.max() == 1. and d.min() == -1.
    d += 1.
    d /= 2.     # 0~1
    d *= 255.   # 0~255
    d = d.astype(np.uint8).astype(np.float)
    d /= 255    # 0~1
    d -= 0.5
    d *= 2.     # -1~1
    np.savez(path_out, **{'data':d,'target':f_in['target']})
