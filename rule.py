from numba import cuda

@cuda.jit(device=True)
def rule(state, alive_neighbors):
    # conway's game of life rule
    if state == 1:
        if alive_neighbors < 2 or alive_neighbors > 3:
            return 0
        else:
            return 1
    else:
        if alive_neighbors == 3:
            return 1
        else:
            return 0