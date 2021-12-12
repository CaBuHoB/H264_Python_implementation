import time

from FileInOut import load_video, save_video
from SchemeRealization import scheme


def main():
    video_names = ['2']

    Hb = 16; Wb = Hb
    Rx = 16; Ry = Rx

    start = time.time()
    for video_name in video_names:
        print('\nProcessing ' + str(video_name))

        W, H, format, codec, rate, frames = load_video(in_video_name=video_name)
        save_video(video_name, W, H, format, codec, rate, frames, save_pics=False)

        num_frames = len(frames)

        start_video = time.time()

        recovered_frames, diff_frames, encoded_frames, _, _, _ = \
        scheme(input_frames=frames, video_length=num_frames, H=H, W=W, Hb=Hb, Wb=Wb, 
            Rx=Rx, Ry=Ry, dx=1, dy=1, Q=1)

        print('Done in ' + str(time.time() - start_video) + 's\n', end='')

        save_video(video_name + '_encoded_',
                W, H, format, codec, rate, encoded_frames, save_pics=False)
        save_video(video_name + '_diff_',
                W, H, format, codec, rate, diff_frames, save_pics=False)
        save_video(video_name + '_decoded_',
                W, H, format, codec, rate, recovered_frames, save_pics=False)

    print('Total processing time: ' + str('{:.3f}'.format(time.time() - start)))


if __name__ == '__main__':
    main()
