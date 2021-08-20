import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time
import random
import numpy as np
import logging

import os

import math as m
import quaternion

from opensimplex import OpenSimplex
import matplotlib.pyplot as plt



class SimplexNoise:
    """
    A simplex action noise
    """
    def __init__(self, dim, s1, s2):
        super().__init__()
        self.idx = 0
        self.dim = dim
        self.s1 = s1
        self.s2 = s2
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return np.array([(self.noisefun.noise2d(x=self.idx / self.s1, y=i*10) + self.noisefun.noise2d(x=self.idx / self.s2, y=i*10)) for i in range(self.dim)])

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

def test_simplex_noise():
    nf = SimplexNoise(1, 30, 300)
    n_samples = 200
    nd = [nf() for _ in range(n_samples)]
    plt.plot(range(n_samples), nd)
    plt.show()

if __name__ == "__main__":
    test_simplex_noise()