import numpy as np
import cv2


def find_inst_bbox(maski, j):
    inst_cr = np.where(maski == j)
    rl = inst_cr[0].min()
    rr = inst_cr[0].max()
    cl = inst_cr[1].min()
    cr = inst_cr[1].max()
    bbox = [rl, cl, rr, cr]
    return bbox

def square_image(X, M=None):
    """
    X: list or nparray
    channel first
    """
    nimg = len(X)
    imgs = []
    masks = []
    for i in range(nimg):
        Ly, Lx = X[i].shape[-2:]
        dxy = Ly - Lx
        xpad1, xpad2, ypad1, ypad2 = 0, 0, 0, 0
        if dxy <= 0:
            img_size = Lx
            dxy = Lx - Ly
            ypad1 = int(dxy//2)
            ypad2 = dxy-ypad1
        elif dxy > 0:
            img_size = Ly
            xpad1 = int(abs(dxy//2))
            xpad2 = abs(dxy)-xpad1
        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, img_size, img_size), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        if M is not None:
            maski = np.pad(M[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            masks.append(maski)

    if M is not None:
        return imgs, masks
    else:
        return imgs

def resize_image(X, M=None, rsc=1.0, xy=None, interpolation=cv2.INTER_LINEAR):
    """
    X: list or nparray
    channel first
    """
    nimg = len(X)
    imgs = []
    masks = []
    for i in range(nimg):
        if xy is None:
            Ly = int(rsc * X[i].shape[-2])
            Lx = int(rsc * X[i].shape[-1])
        else:
            Ly, Lx = xy

        if X[i].ndim == 3:
            nchan = X[i].shape[0]
            imgi = np.zeros((nchan, Ly, Lx), np.float32)
            for m in range(nchan):
                imgi[m] = cv2.resize(X[i][m], (Lx, Ly), interpolation=interpolation)
        elif X[i].ndim == 2:
            imgi = cv2.resize(X[i], (Lx, Ly), interpolation=interpolation)
        imgs.append(imgi)

        if M is not None:
            maski = cv2.resize(M[i], (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            masks.append(maski)

    if M is not None:
        return imgs, masks
    else:
        return imgs

def pad_image(X, M=None, xy=None):
    """
    X: list or nparray
    channel first
    """
    nimg = len(X)
    imgs = []

    if M is not None: masks = []
    for i in range(nimg):
        Ly, Lx = X[0].shape[-2:]
        dy = xy[0] - Ly
        dx = xy[1] - Lx
        ypad1, ypad2, xpad1, xpad2 = 0, 0, 0, 0
        if dy>0:
            ypad1 = int(dy // 2)
            ypad2 = dy - ypad1
        if dx>0:
            xpad1 = int((dx // 2))
            xpad2 = dx - xpad1

        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, xy[0], xy[0]), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        if M is not None:
            maski = np.pad(M[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            masks.append(maski)

    if M is not None:
        return imgs, masks
    else:
        return imgs

def normalize_img(img, axis=-1, invert=False):
    """
    optional inversion

    Parameters
    ------------
    img: ND-array (at least 3 dimensions)
    axis: channel axis to loop over for normalization

    Returns
    img: ND-array, float32
        normalized image of same size
    """
    if img.ndim<3:
        raise ValueError('Image needs to have at least 3 dimensions')

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k])
            if invert:
                img[k] = -1*img[k] + 1
    img = np.moveaxis(img, 0, axis)
    return img

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = img.copy()
    if np.percentile(X, 99) - np.percentile(X, 1) == 0:
        X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1) + 1e-6)  # 这种归一化对0很多的图像不适用
    else:
        X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X
