import cv2, struct, h5py
import numpy as np
from datetime import date
from numpy.random import uniform
from rat_helper import read_rat, clip_rat
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

DATA_FOLDER = '/Volumes/working/Silvan_Aletsch_coregFeb2017/coreg_2011-2018_bscatter_geo/full_scene_amplitude_rat'
EXPORT_FOLDER = '/Volumes/working/Shiyi_MultilookingVelocity/clips'


def open_dat_file(path: str) -> np.ndarray:
    print(f"Opening file: {path}")
    with open(path, 'rb') as rfile:
        rng = struct.unpack('>l', rfile.read(4))[0]
        azm = struct.unpack('>l', rfile.read(4))[0]
        print(f" - Range:   {rng}")
        print(f" - Azimuth: {azm}")
        dt = np.dtype('>c8')
        arr = np.fromfile(rfile, dtype=dt)
        arr = arr.astype(np.complex_)
    return arr.reshape((azm, rng))

class SLC_IMAGE(object):

    def __init__(self, img_filename:str, n_patches=(0, 0)):

        self.data = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

        self.rows, self.cols = self.data.shape

        self.timestamp = date(*map(int, img_filename.split('_')[-1][0:10].split('-')))

        np_row, np_col = n_patches

        if (np_row != 0) & (np_col != 0):
            self.patches = [self.data[np_row*j: np_row*(j+1), np_col*i: np_col*(i+1)]
                                for j in range(int(self.rows / np_row))
                                    for i in range(int(self.cols / np_col))
                            ]
        else:
            if np_col == 0 & np_row != 0:
                self.patches = [self.data[np_row*j: np_row*(j+1), :] for j in range(int(self.rows / np_row))]
            elif np_row == 0 & np_col != 0:
                self.patches = [self.data[:,  np_col*i: np_col*(i+1)] for i in range(int(self.cols / np_col))]
            else:
                self.patches = self.data

    def imshift(self, shifts=(0, 0)):

        M = np.float32([[1, 0, shifts[0]], [0, 1, shifts[1]]])
        dst = cv2.warpAffine(self.data, M, (self.cols, self.rows))

        return dst


def velocity_estimation(img_list, vrange=(0, 10), vorient=(0, np.pi), iteration=40):

    images = []
    for img in img_list:
        images.append(SLC_IMAGE(img))

    vx_unit, vy_unit = np.asarray(images[0].data.shape)
    # print(np.asarray(v_units))
    max_contrast = np.zeros_like(images[0].data)
    velocity_field = np.zeros_like(images[0].data)
    orientation_field = np.zeros_like(images[0].data)
    best_stack = np.zeros_like(images[0].data)

    for i in range(iteration):
        print(f"{'*'*10}Iteration at {i}")
        alp = uniform(vorient[0], vorient[1], 1)
        # v = uniform(vrange[0], vrange[1], 1)
        v = 0
        vx, vy = v * np.array([np.cos(alp), np.sin(alp)])

        ref_img = images[0]
        t0 = ref_img.timestamp

        stack = ref_img.data

        for img in images[1: ]:
            dx, dy = np.array([vx, vy]) * (img.timestamp - t0).days
            shifted_img = img.imshift((dx/vx_unit, dy/vy_unit))
            stack += shifted_img

        # stack = stack / (len(images)-1)

        cv2.imwrite(EXPORT_FOLDER+f'/stack_vx_{vx}_vy_{vy}.png', stack)

        filter_width = 25
        highpass = stack - cv2.GaussianBlur(stack,(filter_width, filter_width), 0)
        weight = cv2.GaussianBlur(np.abs(highpass), (filter_width, filter_width), 0)

        idx = np.where(weight > max_contrast)
        max_contrast[idx] = weight[idx]
        best_stack[idx] = stack[idx]
        velocity_field[idx] = v
        orientation_field[idx] = alp

    return best_stack, velocity_field, orientation_field

if __name__=='__main__':

    ratfile = DATA_FOLDER + '/s0amp_2016-01-13.rat'
    ratdata = read_rat(ratfile)
    print(ratdata[0, 281])

    # row_clip = [6500, 7500]
    # col_clip = [11000, 12000]
    # cliplabel = ratfile.split('/')[-1][0:-4]
    # #
    # # clip_rat(ratfile, row_clip, col_clip, cliplabel, EXPORT_FOLDER+'/'+cliplabel)
    #
    #
    # with h5py.File(EXPORT_FOLDER+'/'+cliplabel+'.h5', 'r') as h5file:
    #     print(h5file.keys())
    #     data = h5file.get(cliplabel)[:]
    # # #
    # print(data.shape)
    # # print(data[1].shape)
    # #
    # plt.imshow(np.abs(data)**2, cmap='gray')
    # plt.show()

    # img = plt.imread('/Volumes/working/Silvan_Aletsch_coregFeb2017/coreg_2011-2018_bscatter_geo/clip_geo_aletsch_gruenegg/s0amp_2011-11-17.tiff')
    #
    # img_2 = plt.imread('/Volumes/working/Silvan_Aletsch_coregFeb2017/coreg_2011-2018_bscatter_geo/clip_geo_aletsch_gruenegg/s0amp_2011-11-28.tiff')
    #
    # print(img_2.shape)
    # # print(img[0:10, 0:10])
    # # xx, yy = np.meshgrid(np.arange(1, img.shape[0]+1), np.arange(1, img.shape[1]+1))
    # # f = interp2d(xx.flatten(), yy.flatten(), img_2.flatten())
    # # img2 = f[xx, yy]
    #
    #
    # plt.imshow(img, cmap='gray')
    #
    # plt.show()

    # arr = read_rat(file_name)
    # print(arr.shape)



    # sp = SLC_IMAGE('/Volumes/working/Silvan_Aletsch_coregFeb2017/coreg_2011-2018_bscatter_geo/clip_geo_aletschgletscher/s0amp_2011-11-06.tiff').data.shape
    #
    # print(sp)
    # version = 1.0
    # img_arr = open_dat_file(file_name)

    # print(img_arr.shape)
    # img_files = glob.glob(DATA_FOLDER+'/*.tiff')[11:20]
    # # print(img_files)
    #
    # test_img = SLC_IMAGE(img_files[0])
    # # print(test_img.data[0:10, 0:10])
    #
    # stack_noshift = np.zeros_like(test_img.data)
    # for img in img_files:
    #     sub = SLC_IMAGE(img).data
    #     # sub_interpo = cv2.resize(10**(sub/10), None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #     stack_noshift += SLC_IMAGE(img).data
    #
    # # cv2.imshow('ref_img', test_img.data)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # noshifted = cv2.GaussianBlur(stack_noshift,(5, 5), 0)
    # cv2.imshow('image', stack_noshift)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #
    # vorient = -45/180 * np.pi

    # beststack, vfield, orientfield = velocity_estimation(img_files, vrange=(0, 1000), vorient=(vorient+0.01, vorient-0.01))

    # cv2.imshow('image', beststack)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # with open(file_name, 'rb') as lun:
    #     magicreal = np.fromfile(file=lun, dtype="i4", count=1)
    #
    # if magicreal != 844382546:  # Check if maybe we have a RAT V1 File...
    #     with open(file_name, 'rb') as lun:
    #         ndim = np.fromfile(file=lun, dtype="<i4", count=1)
    #     xdrflag = 0
    #     if ndim < 0 or ndim > 9:
    #         ndim = ndim.byteswap()
    #         xdrflag = 1
    #     if ndim < 0 or ndim > 9:
    #         print(red + "ERROR: format not recognised!" + endc)
    #     else:
    #         version = 1.0
    # else:  # -------------- Yeah, RAT 2.0 found
    #     with open(file_name, 'rb') as lun:
    #         lun.seek(4)
    #         version = np.fromfile(file=lun, dtype="float32", count=1)[0]
    #     xdrflag = 0
    #
    # print(version, xdrflag, ndim)
    #
    # with open(file_name, 'rb') as lun:
    #     magiclong = lun.read(4)
    #     if magiclong == b'RAT2':
    #         print('true')
    #     else:
    #         print('false')
    #
    # data_type = '>i4'
    # offset = 4 * 4
    # with open(file_name, 'rb') as lun:
    #     ndim = int(np.fromfile(file=lun, dtype=data_type, count=1))
    #     shape = np.fromfile(file=lun, dtype=data_type, count=ndim).tolist()
    #     shape = shape[::-1]
    #     var = int(np.fromfile(file=lun, dtype=data_type, count=1))
    #     rattype = int(np.fromfile(file=lun, dtype=data_type, count=1))
    #     lun.seek(offset, 1)
    #     info = np.fromfile(file=lun, dtype="B", count=80).tostring().rstrip()
    #
    # print(ndim, shape, var, rattype, info)
    #
    # block = np.zeros(2 * len(shape), dtype=np.int)
    # block[1::2] = shape
    # ind = tuple(map(
    #     lambda x, y: slice(x, y, None), block[::2], block[1::2]))
    #
    # offset = int(104 + 4 * ndim + 4 * xdrflag)
    # print(offset)

    # with open(file_name, 'rb') as lun:
    #     mm = mmap.mmap(
    #         lun.fileno(), length=0, access=mmap.ACCESS_READ)
    #
    #     arr = (np.ndarray.__new__(np.ndarray, shape, dtype=np.float32,
    #                               buffer=mm, offset=offset)[ind])
    #     if xdrflag == 1:
    #         arr = arr.byteswap()
    # arr_new = np.zeros(shape=arr.shape, dtype=arr.dtype)
    # arr_new[:] = arr
    #
    # print(arr_new.shape)