import numpy as np
from numba import jit


N = 8
cos_for_DCT = np.zeros(shape=(N, N, N, N))
for k in range(N):
    for l in range(N):
        for j in range(N):
            for i in range(N):
                cos_for_DCT[k][l][j][i] = \
                    np.cos((2 * j + 1) * np.pi / (2 * N) * k) * \
                    np.cos((2 * i + 1) * np.pi / (2 * N) * l)

sqrt_C_0 = np.sqrt(1.0 / N)
sqrt_C_else = np.sqrt(2.0 / N)

x_bypass = np.array([1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4,
                     3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3,
                     2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2,
                     3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7])
y_bypass = np.array([0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1,
                     2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4,
                     5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7,
                     7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7])


@jit(nopython=True, cache=True)
def add_remove_zero_rows_columns(frame, real_H, real_W, add=True):
    if add:
        H_ = real_H if real_H % N == 0 else np.ceil(real_H / N) * N
        W_ = real_W if real_W % N == 0 else np.ceil(real_W / N) * N
    else:
        H_ = real_H; W_ = real_W

    if len(frame) == H_ and len(frame[0]) == W_:
        return frame.astype(np.int64)

    modified_frame = np.zeros(shape=(int(H_), int(W_)), dtype=np.int64)
    for j in range(H_):
        for i in range(W_):
            modified_frame[j][i] = frame[j][i]
    return modified_frame


@jit(nopython=True, cache=True)
def DCT(frame):
    modified_frame = frame 
    dct_frame = np.zeros(modified_frame.shape, dtype=np.int64)
    for y in np.arange(0, len(modified_frame), 8):
        for x in np.arange(0, len(modified_frame[0]), 8):
            block = np.zeros((N, N), dtype=np.int64)
            for k in range(N):
                for l in range(N):
                    Ck_sqrt = sqrt_C_0 if k == 0 else sqrt_C_else
                    Cl_sqrt = sqrt_C_0 if l == 0 else sqrt_C_else
                    sum = 0
                    for j in range(N):
                        for i in range(N):
                            sum += modified_frame[j + y][i + x] * \
                                    cos_for_DCT[k][l][j][i]
                    block[k][l] = np.round(Ck_sqrt * Cl_sqrt * sum)
            for j in range(N):
                for i in range(N):
                    dct_frame[j + y][i + x] = block[j][i]
    return dct_frame


@jit(nopython=True, cache=True)
def inverse_DCT(frame, H, W):
    inverse_DCT_frame = np.zeros((len(frame), len(frame[0])), dtype=np.int64)
    for y in np.arange(0, len(frame), 8):
        for x in np.arange(0, len(frame[0]), 8):
            block = np.zeros((N, N))
            for j in range(N):
                for i in range(N):
                    sum = 0
                    for k in range(N):
                        for l in range(N):
                            Ck_sqrt = sqrt_C_0 if k == 0 else sqrt_C_else
                            Cl_sqrt = sqrt_C_0 if l == 0 else sqrt_C_else
                            sum += Ck_sqrt * Cl_sqrt * frame[k + y][l + x] * \
                                    cos_for_DCT[k][l][j][i]
                    block[j][i] = np.round(sum)
            for j in range(N):
                for i in range(N):
                    inverse_DCT_frame[j + y][i + x] = block[j][i]
    return inverse_DCT_frame


@jit(nopython=True, cache=True)
def Q(frame, R_for_Q):
    Q_luma = np.zeros(shape=(N, N), dtype=np.int64)
    for j in range(N):
        for i in range(N):
            Q_luma[j][i] = 1 + (i + j) * R_for_Q

    modified_frame = np.zeros((len(frame), len(frame[0])), dtype=np.int64)
    for y in np.arange(0, len(frame), 8):
        for x in np.arange(0, len(frame[0]), 8):
            for j in range(N):
                for i in range(N):
                    modified_frame[j + y][i + x] = \
                        np.round(frame[j + y][i + x] / Q_luma[j][i])
    return modified_frame


@jit(nopython=True, cache=True)
def inverse_Q(frame, R_for_Q):
    Q_luma = np.zeros(shape=(N, N), dtype=np.int64)
    for j in range(N):
        for i in range(N):
            Q_luma[j][i] = 1 + (i + j) * R_for_Q

    modified_frame = np.zeros((len(frame), len(frame[0])))
    for y in np.arange(0, len(frame), 8):
        for x in np.arange(0, len(frame[0]), 8):
            for j in range(N):
                for i in range(N):
                    modified_frame[j + y][i + x] = \
                        np.round(frame[j + y][i + x] * Q_luma[j][i])
    return modified_frame


@jit(nopython=True, cache=True)
def entropy(vector: np.ndarray):
    max_num = np.max(np.array([vector.max(), np.abs(vector.min())])) + 1
    p = np.zeros(int(max_num) * 2, dtype=np.int64)

    for i in range(len(vector)):
        num = int(vector[i])
        p[num + max_num] += 1
    p = p / p.sum()

    entropy_num = 0.0
    for i in range(max_num):
        if p[i] != 0:
            entropy_num += p[i] * np.log2(1/p[i])
    return entropy_num


@jit(nopython=True, cache=True)
def get_num_bits_for_encoding(matrix):
    H = round(matrix.shape[0] / 8)
    W = round(matrix.shape[1] / 8)
    DC = np.zeros(H * W, dtype=np.int64)
    for i in range(H):
        for j in range(W):
            DC[i * W + j] = matrix[i * 8][j * 8]
    DC_previous = np.round(DC.sum() / DC.size)
    delta_DC = np.zeros(DC.size, dtype=np.float32)
    delta_DC[0] = DC[0] - DC_previous
    for i in range(1, len(DC)):
        delta_DC[i] = DC[i] - DC[i - 1]

    coded_bit_categories = [(np.ceil(np.log2(np.abs(np.round(DC)) + 1)),
                np.uint8(~int(DC) if DC < 0 else DC)) for DC in delta_DC]

    index_i = np.arange(0, matrix.shape[0], 8)
    index_j = np.arange(0, matrix.shape[1], 8)
    all_pair_run_level = []
    for i in range(len(index_i)):
        for j in range(len(index_j)):
            run_level = []
            count = 0
            for k in range(len(x_bypass)):
                AC = int(matrix[index_i[i] + 
                    y_bypass[k]][index_j[j] + 
                        x_bypass[k]])
                if AC != 0:
                    if count > 15:
                        run_level.append((15, 0))
                        count -= 16
                    run_level.append((count, AC))
                    count = 0
                else:
                    count += 1

            if count > 0:
                run_level.append((0, 0))
            all_pair_run_level.append(run_level)

    all_pair_bc_level_magnitude_level = []
    for pair_run_level in all_pair_run_level:
        pair_bc_level_magnitude_level = []
        for pair in pair_run_level:
            BC = np.uint8(np.ceil(np.log2(np.abs(np.round(pair[1])) + 1)))
            magnitude = np.uint8(~pair[1] if pair[1] < 0 else pair[1])
            pair_bc_level_magnitude_level.append((BC, magnitude))
        all_pair_bc_level_magnitude_level.append(pair_bc_level_magnitude_level)

    all_triple_run_bc_level_magnitude_level = []
    for i in range(len(all_pair_run_level)):
        triple_run_bc_level_magnitude_level = []
        for j in range(len(all_pair_run_level[i])):
            TRIPLE = (all_pair_run_level[i][j][0],
                      all_pair_bc_level_magnitude_level[i][j][0],
                      all_pair_bc_level_magnitude_level[i][j][1])

            triple_run_bc_level_magnitude_level.append(TRIPLE)
        all_triple_run_bc_level_magnitude_level.append( \
                    triple_run_bc_level_magnitude_level)

    BC_delta_DC = np.zeros(len(coded_bit_categories), dtype=np.int32)
    for i in range(len(coded_bit_categories)):
        BC_delta_DC[i] = coded_bit_categories[i][0]

    run = []
    for pair_run_level in all_pair_run_level:
        for pair in pair_run_level:
            run.append(pair[0])

    BC_level = []
    for pair_bc_level_magnitude_level in all_pair_bc_level_magnitude_level:
        for pair in pair_bc_level_magnitude_level:
            BC_level.append(pair[0])

    BC_bits_size = np.array([1, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
                                5, 6, 6, 6, 7, 8, 8])
    MG_bits_size = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                11, 12, 13, 14, 15, 16, 16])
    run_bits_size = 4

    all_size = 0
    for i in range(len(BC_delta_DC)):
        BC_category = BC_delta_DC[i]
        all_size += BC_bits_size[BC_category] + MG_bits_size[BC_category]
    all_size += run_bits_size * len(run)
    for i in range(len(BC_level)):
        BC_category = BC_level[i]
        all_size += BC_bits_size[BC_category] + MG_bits_size[BC_category]

    return all_size


@jit(nopython=True, cache=True)
def PSNR(signal1, signal2):
    numerator = len(signal1) * len(signal1[0]) * (2 ** 8 - 1) ** 2
    denominator = 0
    for j in range(len(signal1)):
        for i in range(len(signal1[0])):
            denominator += (signal1[j][i] - signal2[j][i]) ** 2
    if denominator == 0: return np.inf
    return 10 * np.log10(numerator / denominator)


@jit(nopython=True, cache=True)
def form_frame_on_ME_vectors(base, H, W, Hb, Wb, vectors):
    new_frame = np.zeros((H, W), dtype=np.int64)
    for j in range(int(np.ceil(H / Hb))):
        for i in range(int(np.ceil(W / Wb))):
            delta_x = vectors[j][i][0]
            delta_y = vectors[j][i][1]
            hb = Hb if j * (Hb + 1) <= H else H - j * Hb
            wb = Wb if i * (Wb + 1) <= W else W - i * Hb
            for y in range(hb):
                for x in range(wb):
                    h = j * Hb + y
                    w = i * Wb + x
                    Y = base[h + int(delta_y)][w + int(delta_x)]
                    new_frame[h][w] = Y
    return new_frame
