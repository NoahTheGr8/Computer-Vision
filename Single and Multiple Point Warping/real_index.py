import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def real_index(Source, rows, cols):
    rows_floor = (rows).astype(int)
    rows_ceil = np.ceil(rows).astype(int)
    cols_floor = (cols).astype(int)
    cols_ceil = np.ceil(cols).astype(int)

    wr0 = (1 - rows + rows_floor)
    wc0 = (1 - cols + cols_floor)

    if len(Source.shape) == 3: # To make it work with color images
        wr0 = np.expand_dims(wr0,axis=2)
        wc0 = np.expand_dims(wc0,axis=2)

    wr1 = 1 - wr0
    wc1 = 1 - wc0

    Dest = Source[rows_floor,cols_floor]*wr0*wc0 + Source[rows_floor,cols_ceil]*wr0*wc1 + Source[rows_ceil,cols_floor]*wr1*wc0 + Source[rows_ceil,cols_ceil]*wr1*wc1

    return Dest

