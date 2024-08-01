import matplotlib.pyplot as plt
import cv2
import io
import numpy as np
import glob
import os

def render_frame(grid, filename):
    # plt.figure(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
    # plt.imshow(grid, cmap='binary')
    # plt.axis('off')
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
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