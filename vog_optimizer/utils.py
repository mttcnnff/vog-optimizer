from typing import List

import numpy as np


def build_image_grid(imgs: List[List]) -> np.ndarray:
    img_rows = [np.hstack([np.array(img) for img in row]) for row in imgs]
    return np.vstack(img_rows)
