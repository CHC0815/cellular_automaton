from numba import cuda
import numpy as np
from numba import cuda
import math
from tqdm import tqdm
import cv2
from src.renderer import render_frame, create_video, VideoRenderer
from rule import rule
from PIL import Image
import time

@cuda.jit
def cellular_automaton_kernel(current_grid, next_grid, neighborhood):
    x, y = cuda.grid(2)
    if x >= current_grid.shape[0] or y >= current_grid.shape[1]:
        return
    
    alive_neighbors = 0
    for dx, dy in neighborhood:
        nx, ny = x + dx, y + dy
        if 0 <= nx < current_grid.shape[0] and 0 <= ny < current_grid.shape[1]:
            alive_neighbors += current_grid[nx, ny]

    next_grid[x, y] = rule(current_grid[x, y], alive_neighbors)


def automaton(iterations:int = 100, s: int = 1024, interactive: bool = False, save: bool = False, show: bool = True, initial_state: np.ndarray = None):
    if initial_state is not None:
        grid = initial_state
    else:
        grid = np.random.randint(2, size=(s, s)).astype(np.int8)
    new_grid = np.zeros_like(grid)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(grid.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(grid.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    moore_neighborhood = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    von_neumann_neighborhood = [(0, -1), (-1, 0), (1, 0), (0, 1)]

    d_neighborhood = cuda.to_device(moore_neighborhood)

    if save:
        video = VideoRenderer('cellular_automaton.mp4', grid.shape[1], grid.shape[0])

    for i in tqdm(range(iterations)):
        d_grid = cuda.to_device(grid)
        d_new_grid = cuda.to_device(new_grid)
        
        cellular_automaton_kernel[blockspergrid, threadsperblock](d_grid, d_new_grid, d_neighborhood)
        d_new_grid.copy_to_host(new_grid)
        
        grid, new_grid = new_grid, grid

        frame = video.render_frame(grid, save=save)

        if show:
            cv2.imshow('frame', frame)
        if show and interactive:
            cv2.waitKey(0)
        elif show and not interactive:
            cv2.waitKey(10)

    if save:
        video.finish()
    print("Simulation complete")