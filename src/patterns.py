def clock_pattern():
    return [
        (9, 8), (9, 9), (9, 10),
        (10, 8), (10, 9), (10, 10),
        (11, 8), (11, 9), (11, 10)
    ]

def f_pentominio_pattern():
    return [
        (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)
    ]


def apply_pattern(x, y, pattern, grid):
    for (dx, dy) in pattern:
        grid[x + dx, y + dy] = 1