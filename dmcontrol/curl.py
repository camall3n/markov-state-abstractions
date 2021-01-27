import numpy as np

from skimage.util import view_as_windows
import numpy as np

def random_crop(imgs, output_width=84):
    """Vectorized random crop
    args:
        imgs: shape (B,C,H,W)
        output_width: output size (e.g. 84)
    """
    n = imgs.shape[0]  # batch size
    input_width = imgs.shape[-1]  # e.g. 100
    crop_max = input_width - output_width

    imgs = np.transpose(imgs, (0, 2, 3, 1))

    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)

    # creates all sliding window
    # combinations of size (output_width)
    windows = view_as_windows(imgs, (1, output_width, output_width, 1))[..., 0, :, :, 0]

    # selects a random window
    # for each batch element
    cropped = windows[np.arange(n), w1, h1]
    return cropped

def center_crop(imgs, output_width=84):
    """Vectorized center crop
    args:
        imgs: shape (B,C,H,W)
        output_width: output size (e.g. 84)
    """
    n = imgs.shape[0]  # batch size
    input_width = imgs.shape[-1]  # e.g. 100
    crop_max = input_width - output_width

    imgs = np.transpose(imgs, (0, 2, 3, 1))

    w1 = crop_max // 2
    h1 = crop_max // 2

    # creates all sliding window
    # combinations of size (output_width)
    windows = view_as_windows(imgs, (1, output_width, output_width, 1))[..., 0, :, :, 0]

    # selects a random window
    # for each batch element
    cropped = windows[np.arange(n), w1, h1]
    return cropped