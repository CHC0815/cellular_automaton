import matplotlib.pyplot as plt
import cv2

def render_frame(grid, filename):
    plt.imshow(grid, cmap='binary')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_video(frames, output_filename):
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'avc1'), 10, (width, height)) #mp4v

    for frame in frames:
        video.write(cv2.imread(frame))

    cv2.destroyAllWindows()
    video.release()