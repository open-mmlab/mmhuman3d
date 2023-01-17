import numpy as np
import skimage.transform
from PIL import Image


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        t = np.dot(get_rot_transf(res, rot), t)
    return t


def get_rot_transf(res, rot):
    """Generate rotation transformation matrix."""
    if rot == 0:
        return np.identity(3)
    rot = -rot  # To match direction of rotation from cropping
    rot_mat = np.zeros((3, 3))
    rot_rad = rot * np.pi / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Need to rotate around center
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    rot_transf = np.dot(t_inv, np.dot(rot_mat, t_mat))
    return rot_transf


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = skimage.transform.rotate(new_img, rot).astype(np.uint8)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img_resized = np.array(
        Image.fromarray(new_img.astype(np.uint8)).resize(res))
    return new_img_resized, new_img, new_shape


def flip_img(img):
    """Flip rgb images or masks.

    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img
