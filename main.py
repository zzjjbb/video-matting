import sys
import numpy as np
import av
import matplotlib.pyplot as plt
import logging
from refine import process, calc_mot


logging.basicConfig(format='%(asctime)s [%(levelname).1s] %(message)s', level=logging.DEBUG)


def read_all(video_name, pix_fmt='rgb24'):
    with av.open(video_name) as v:
        metadata = {
            'fps': v.streams[0].rate,
            'width': v.streams[0].width,
            'height': v.streams[0].height
        }
        frames = []
        for f in v.decode():
            frames.append(f.to_ndarray(format=pix_fmt))
            if len(frames) == MAX_FRAMES:
                break
        frames = np.stack(frames)
    return frames, metadata

def name_suffix(name, suffix=''):
    name = name.split('.')
    name[-2] += '_' + suffix
    return '.'.join(name)

MAX_FRAMES = 300
input_name = sys.argv[1]
matte_name = name_suffix(input_name, "matte")

print('read')
in_frames, metadata = read_all(input_name)
matte_frames, _ = read_all(matte_name, pix_fmt='gray8')

print('read end')

# out_frames = all_frames
out_frames, err_frames = process(in_frames, matte_frames, motion=False)

output_name, err_name = name_suffix(input_name, "motionmatte"), name_suffix(input_name, "error")
print('write')
metadata['fps'] = 30
with av.open(output_name, 'w') as output:
    stream = output.add_stream('libx264', options={'preset': 'fast', 'crf': '17'}, rate=metadata['fps'])
    stream.width = metadata['width']
    stream.height = metadata['height']
    stream.pix_fmt = 'gray'
    for f in out_frames:
        packets = stream.encode(av.VideoFrame.from_ndarray(f, format='gray'))
        output.mux(packets)
    output.mux(stream.encode())

with av.open(err_name, 'w') as output:
    stream = output.add_stream('libx264rgb', options={'preset': 'fast', 'crf': '17'}, rate=metadata['fps'])
    stream.width = metadata['width']
    stream.height = metadata['height']
    stream.pix_fmt = 'rgb24'
    for f in err_frames:
        packets = stream.encode(av.VideoFrame.from_ndarray(f, format='rgb24'))
        output.mux(packets)
    output.mux(stream.encode())
print('write end')
