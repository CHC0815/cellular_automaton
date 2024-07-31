import numpy as np
from src.automaton import automaton
from src.patterns import *

def main():
    size = 1024
    inital_state = np.zeros(shape=(size, size), dtype=np.int8)
    apply_pattern(512, 512, gosper_glider_gun_pattern(), inital_state)
    automaton(iterations=1000, s=size, interactive=False, show=True, save=False, initial_state=inital_state)

if __name__ == '__main__':
    main()