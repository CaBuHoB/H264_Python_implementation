import av
import numpy as np

from PIL import Image


def load_video(in_video_name):
    input = av.open('Video/' + str(in_video_name) + '.avi')

    stream = input.streams.video[0]
    W = stream.width
    H = stream.height
    format = stream.pix_fmt
    codec = stream.codec_context.name
    rate = stream.average_rate

    frames = []
    for frame in input.decode(stream):
        Y = np.array(frame.to_image().convert('YCbCr'))[:,:,0]
        frames.append(Y)

    input.close()

    return W, H, format, codec, rate, frames


def save_video(video_name, W, H, format, codec, rate, frames, save_pics=False):
    output = av.open('Results/Video/' + str(video_name) + '.avi', mode='w')
    
    stream_out = output.add_stream(codec, rate=rate)
    stream_out.width = W
    stream_out.height = H
    stream_out.pix_fmt = format

    for n, frame in enumerate(frames):
        i = Image.fromarray(frame.astype(np.uint8)).convert('RGB')

        if save_pics:
            i.save('Results/Images/' + str(video_name) + str(n) + '.bmp', 'bmp')

        image = av.VideoFrame.from_image(i)
        for packet in stream_out.encode(image):
            output.mux(packet)

    output.close()
