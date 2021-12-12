import numpy as np
import SchemeBlocks as sb
from Vectors import motion_estimation


def clip(num):
    return 0 if num < 0 else 255 if num > 255 else num


def scheme(input_frames, video_length, H, W, Hb, Wb, Rx, Ry, dx, dy, Q):
    clip_np = np.vectorize(clip)
    Enc_1 = 0; Enc_2 = 0
    recovered_frames = []; diff_frames = []; encoded_frames = []
    vectors = np.zeros(video_length, dtype=object)
    psnr = 0

    for f in range(0, video_length):
        print('\r' + str(f + 1) + '/' + str(video_length) + ' frame', end='')
        cur = input_frames[f]

        if len(recovered_frames) == 0:
            diff_frame = cur
            new_frame = 0
        else:
            base = recovered_frames[-1]
            cur_vectors = motion_estimation(cur, base, H, W, Hb, Wb, 
                                            Rx, Ry, dx, dy)
            vectors[f] = cur_vectors
            BC = np.round(np.max([Rx, Ry])) + 1
            Enc_1 += cur_vectors.size * np.ceil(np.log2(BC))
            new_frame = sb.form_frame_on_ME_vectors(base, H, W, 
                                                Hb, Wb, cur_vectors)
            diff_frame = cur - new_frame
            diff_frames.append(clip_np(diff_frame + 128))

        encoded_frame = sb.Q(sb.DCT(diff_frame), Q)
        encoded_frames.append(encoded_frame)
        Enc_2 += sb.get_num_bits_for_encoding(encoded_frame)

        decoded_frame = sb.inverse_DCT(sb.inverse_Q(encoded_frame, Q), H, W)

        recovered_frame = clip_np(decoded_frame + new_frame)
        recovered_frames.append(recovered_frame)

        psnr += sb.PSNR(recovered_frame, cur)

    psnr /= video_length
    coef = (Enc_1 + Enc_2) / (len(np.ravel(input_frames))*8)

    return recovered_frames, diff_frames, encoded_frames, vectors, psnr, coef
