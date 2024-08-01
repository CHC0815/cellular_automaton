import matplotlib.pyplot as plt
import cv2
import io
import numpy as np
import glob
import os

def render_frame(grid, filename):
    cv2.imwrite(filename, np.uint8(grid)*255)


def create_video(frames, output_filename):
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'avc1'), 60, (width, height)) #mp4v

    for frame in frames:
        video.write(cv2.imread(frame))

    cv2.destroyAllWindows()
    video.release()
    _clear_tmp()

def _clear_tmp():
    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)


class VideoRenderer:
    def __init__(self, output_file, width, height):
        self.output_file = output_file
        self.width = width
        self.height = height

        self.video = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'avc1'), 60, (self.width, self.height))

    def render_frame(self, grid, save: bool = True):
        img_rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        img_rgb[grid == 1] = [0,255,0]
        img_rgb[grid == 0] = [0,0,0]
        if save:
            self.video.write(img_rgb)
        return img_rgb
    
    def finish(self):
        self.video.release()