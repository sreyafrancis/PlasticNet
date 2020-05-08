import random
import numpy as np
import pandas as pd


def produce_cloud_batch(size :int = 256, difficulty=1, eccentricity: int = random.randint(0, 4),
                 rot_ecc: int = random.randint(0, 3),
                 rot_mag: float = np.random.beta(2, 5),
                 shear_ecc: int = random.randint(0, 4),
                 shear_mag = random.randint(0, 3),
                 smoothness=6):

    batch = []
    return batch
