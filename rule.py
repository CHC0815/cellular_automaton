from numba import cuda

alive = 1
dead = 0

@cuda.jit(device=True)
def rule(state, alive_neighbors):
    # conway's game of life rule
    if state == alive:
        if alive_neighbors < 2 or alive_neighbors > 3:
            return dead
        else:
            return alive
    else:
        if alive_neighbors == 3:
            return alive
        else:
            return dead