import numpy as np
import mmap, glob, os, h5py


def read_rat(filepath: str) -> np.ndarray:
    """
    Read *.rat file.
    :param filepath:
    :return: numpy array of images
    """
    with open(filepath, 'rb') as lun:
        ndim = int(np.fromfile(file=lun, dtype='>i4', count=1))
        shape = np.fromfile(file=lun, dtype='>i4', count=ndim).tolist()
        shape = shape[::-1]
        block = np.zeros(2 * len(shape), dtype=np.int)
        block[1::2] = shape
        ind = tuple(map(lambda x, y: slice(x, y, None), block[::2], block[1::2]))
        offset = int(104 + 4 * ndim)
        mm = mmap.mmap(lun.fileno(), length=0, access=mmap.ACCESS_READ)
        arr = (np.ndarray.__new__(np.ndarray, shape, dtype=np.float32, buffer=mm, offset=offset)[ind])

    arr_new = np.zeros(shape=arr.shape, dtype=arr.dtype)
    arr_new[:] = arr

    return arr_new


def clip_rat(ratfile: str, row_clip: list, col_clip: list, datalabel: str, save_to: str):

    rat_img = read_rat(ratfile)

    clip_rat_img = rat_img[row_clip[0]:row_clip[1], col_clip[0]: col_clip[1]]

    print(clip_rat_img.shape)

    if save_to.split('.')[-1] != 'h5':
        save_to += '.h5'

    with h5py.File(save_to, 'w') as h5file:
        h5file.create_dataset(datalabel, data=clip_rat_img)

    print(f"Data is saved in DB={datalabel} of HDF5={save_to.split('/')[-1]}.")
    return None

