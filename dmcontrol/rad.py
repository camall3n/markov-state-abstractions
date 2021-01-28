import numpy as np

# Image augmentations from RAD paper: https://arxiv.org/pdf/2004.14990.pdf
# Code from here: https://github.com/MishaLaskin/rad/blob/1246bfd6e716669126e12c1f02f393801e1692c1/data_augs.py
def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):

        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def center_crop_images(images, output_size=(84, 84)):
    h, w = images.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    images = images[:, :, top:top + new_h, left:left + new_w]
    return images

def center_crop_image(image, output_size=84):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs

def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1:h1 + h, w1:w1 + w] = image
    return outs
