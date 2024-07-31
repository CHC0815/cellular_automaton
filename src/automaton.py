from numba import cuda, jit
import numpy as np
from numba import cuda
import math
from tqdm import tqdm
import cv2

from src.renderer import render_frame, create_video

@cuda.jit
def cellular_automaton_kernel(current_grid, next_grid):
    x, y = cuda.grid(2)
    if x >= current_grid.shape[0] or y >= current_grid.shape[1]:
        return
    
    alive_neighbors = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= x + 1 < current_grid.shape[0] and 0 <= y +j < current_grid.shape[1]:
                alive_neighbors += current_grid[x + i, y + j]

    if current_grid[x, y] == 1:
        if alive_neighbors < 2 or alive_neighbors > 3:
            next_grid[x, y] = 0
        else:
            next_grid[x, y] = 1
    else:
        if alive_neighbors == 3:
            next_grid[x, y] = 1
        else:
            next_grid[x, y] = 0

def automaton(n: int = 1024, interactive: bool = False):
    grid = np.random.randint(2, size=(n, n)).astype(np.int32)
    new_grid = np.zeros_like(grid)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(grid.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(grid.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    frames = []
    iterations = 100
    for i in tqdm(range(iterations)):
        d_grid = cuda.to_device(grid)
        d_new_grid = cuda.to_device(new_grid)
        
        cellular_automaton_kernel[blockspergrid, threadsperblock](d_grid, d_new_grid)
        
        d_new_grid.copy_to_host(new_grid)
        
        grid, new_grid = new_grid, grid  # Swap the grids
    
        frame_filename = 'tmp/frame_{num:{fill}{width}}.png'.format(num=i, fill='0', width=int(math.log10(iterations)) + 1)
        render_frame(grid, frame_filename)
        frames.append(frame_filename)

        if interactive:
            cv2.imshow('frame', cv2.imread(frame_filename))
            cv2.waitKey(0)

    create_video(frames, 'cellular_automaton.mp4')
    print("Simulation complete")