import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def L1(cur, base, i, j, Hb, Wb, delta_x, delta_y):
    res = 0
    for y in range(Hb):
        for x in range(Wb):
            h = j + y
            w = i + x
            res += np.abs(cur[h][w] - base[h + int(delta_y)][w + int(delta_x)])
    return res


@jit(nopython=True, cache=True)
def search_in_R(cur, base, H, W, i, j, dx, dy, 
                Rx_start, Rx_end, Ry_start, Ry_end, Wb, Hb, wb, hb):
    delta_x = 0
    delta_y = 0
    minimum = np.inf
    for x in np.arange(Rx_start, Rx_end, dx):
        if Wb*i + x < 0 or Wb*i + wb + x > W: continue
        for y in np.arange(Ry_start, Ry_end, dy):
            if Hb*j + y < 0 or Hb*j + hb + y > H: continue
            estimation = L1(cur, base, i*Wb, j*Hb, hb, wb, x, y)
            if estimation < minimum:
                minimum = estimation
                delta_x = x
                delta_y = y
    return delta_x, delta_y


@jit(nopython=True, cache=True)
def motion_estimation(cur, base, H, W, Hb, Wb, Rx, Ry, dx, dy):
    vectors = np.zeros((int(np.ceil(H / Hb)), int(np.ceil(W / Wb)), 2))

    for j in range(int(np.ceil(H / Hb))):
        for i in range(int(np.ceil(W / Wb))):
            hb = Hb if j * (Hb + 1) <= H else H - j * Hb
            wb = Wb if i * (Wb + 1) <= W else W - i * Hb

            delta_x, delta_y = search_in_R(cur, base, H, W, i, j,
                                                   dx, dy,
                                                   -Rx, Rx,
                                                   -Ry, Ry,
                                                   Wb, Hb, wb, hb)

            vectors[j][i][0] = int(delta_x)
            vectors[j][i][1] = int(delta_y)

    return vectors
