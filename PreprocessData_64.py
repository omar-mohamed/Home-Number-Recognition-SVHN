# These are all the modules we'll be using later.
from __future__ import print_function
import numpy as np

from six.moves import cPickle as pickle
from six.moves import range

import os
import sys
import tarfile
from IPython.display import display, Image

import h5py

import matplotlib.pyplot as plt

from PIL import Image
import random

train_folders = './train/digitStruct.mat'
test_folders = './test/digitStruct.mat'
extra_folders = './extra/digitStruct.mat'

train_dataset = h5py.File(train_folders, "r")
test_dataset = h5py.File(test_folders, "r")
extra_dataset = h5py.File(extra_folders, "r")


# The DigitStructFile is just a wrapper around the h5py data.  It basically references
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    # getName returns the 'name' string for for the n(th) digitStruct.
    def getName(self, n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    # bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox.
    def bboxHelper(self, attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    # getBbox returns a dict of data for the n(th) bbox.
    def getBbox(self, n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self, n):
        s = self.getBbox(n)
        s['name'] = self.getName(n)
        return s

    # getAllDigitStructure returns all the digitStruct from the input file.
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

    # Return a restructured version of the dataset (one structure by boxed digit).
    #
    #   Return a list of such dicts :
    #      'filename' : filename of the samples
    #      'boxes' : list of such dicts (one by digit) :
    #          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
    #          'left', 'top' : position of bounding box
    #          'width', 'height' : dimension of bounding box
    #
    # Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            print("getAllDigitStructure_ByDigit iteration: " + str(i) + " from: " + str(len(pictDat)) + "\n", end=' ',
                  flush=True)
            item = {'filename': pictDat[i]["name"]}
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label'] = pictDat[i]['label'][j]
                figure['left'] = pictDat[i]['left'][j]
                figure['top'] = pictDat[i]['top'][j]
                figure['width'] = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result


train = DigitStructFile(train_folders)
train_data = train.getAllDigitStructure_ByDigit()

test = DigitStructFile(test_folders)
test_data = test.getAllDigitStructure_ByDigit()

extra = DigitStructFile(extra_folders)
extra_data = extra.getAllDigitStructure_ByDigit()


class Dataset:
    """crop images and save them to numpy ndarray"""

    def __init__(self, digitStruct, folder):
        self.digitStruct = digitStruct
        self.folder = folder

    def setDataset(self):
        self.dataset = np.ndarray(shape=(len(self.digitStruct), 64, 64), dtype='float32')

        # 1 length + 5 labels, 0 represents none
        self.labels = np.ones(shape=(len(self.digitStruct), 6), dtype='int') * 10

    def getDataset(self):

        self.setDataset()

        for i in range(len(self.digitStruct)):
            print("getDataset iteration: " + str(i) + " from: " + str(len(self.digitStruct)) + "\n", end=' ',
                  flush=True)
            fin = os.path.join(self.folder, self.digitStruct[i]['filename'])
            im = Image.open(fin)

            boxes = self.digitStruct[i]['boxes']

            if len(boxes) > 5:
                print(fin, "has more than 5 digits")
            else:
                self.labels[i, 0] = len(boxes)
                self.labels[i, 1:len(boxes) + 1] = [int(j['label']) for j in boxes]

            left = [j['left'] for j in boxes]
            top = [j['top'] for j in boxes]
            height = [j['height'] for j in boxes]
            width = [j['width'] for j in boxes]

            box = self.img_box(im, left, top, height, width)

            size = (64, 64)
            region = im.crop(box).resize(size)
            region = self.normalization(region)
            #             print(region.shape)
            self.dataset[i, :, :] = region[:, :]

        print('dataset:', self.dataset.shape)
        print('labels:', self.labels.shape)
        return self.dataset, self.labels

    def img_box(self, im, left, top, height, width):

        im_left = min(left)
        im_top = min(top)
        im_height = max(top) + max(height) - im_top
        im_width = max(left) + max(width) - im_left

        im_top = im_top - im_height * 0.05  # a bit higher
        im_left = im_left - im_width * 0.05  # a bit wider
        im_bottom = min(im.size[1], im_top + im_height * 1.05)
        im_right = min(im.size[0], im_left + im_width * 1.05)

        return (im_left, im_top, im_right, im_bottom)

    def normalization(self, img):
        im = self.rgb2gray(img)  # RGB to greyscale
        pixel_depth=255.0
        # mean = np.mean(im, dtype='float32')
        # std = np.std(im, dtype='float32', ddof=1)
        # new_im=(im - mean) / std
        return (np.array(im, dtype='float32') - (pixel_depth / 2)) / (pixel_depth / 2)

    def rgb2gray(self, img):
        return np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])


test_dataset = Dataset(test_data, 'test')
test_dataset, test_labels = test_dataset.getDataset()

train_dataset = Dataset(train_data, 'train')
train_dataset, train_labels = train_dataset.getDataset()

plt.imshow(train_dataset[29929, :, :])
print(train_labels[29929])

train_dataset = np.delete(train_dataset, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)

extra_dataset = Dataset(extra_data, 'extra')
extra_dataset, extra_labels = extra_dataset.getDataset()


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
extra_dataset, extra_labels = randomize(extra_dataset, extra_labels)

extra_to_training = 150000
extra_to_test = 50000

# from extra take 150k to training
train_dataset = np.append(train_dataset, extra_dataset[:extra_to_training - 1], 0)
train_labels = np.append(train_labels, extra_labels[:extra_to_training - 1], 0)

print('train_dataset:', train_dataset.shape)
print('train_labels:', train_labels.shape)

# from extra take 50k to test
test_dataset = np.append(test_dataset, extra_dataset[extra_to_training:extra_to_training + extra_to_test - 1], 0)
test_labels = np.append(test_labels, extra_labels[extra_to_training:extra_to_training + extra_to_test - 1], 0)

print('test_dataset:', test_dataset.shape)
print('test_labels:', test_labels.shape)

# from extra take 2355 to validation
valid_dataset = extra_dataset[extra_to_training + extra_to_test:]
valid_labels = extra_labels[extra_to_training + extra_to_test:]

print('valid_dataset:', valid_dataset.shape)
print('valid_labels:', valid_labels.shape)

del extra_dataset, extra, extra_data, extra_labels


def disp_sample_dataset(dataset, label):
    items = random.sample(range(dataset.shape[0]), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(label[i][1:5])
        plt.imshow(dataset[i, :, :])
    plt.savefig('./output_images/plt.png')
    plt.show()


disp_sample_dataset(train_dataset, train_labels)

disp_sample_dataset(test_dataset, test_labels)

disp_sample_dataset(valid_dataset, valid_labels)

pickle_file = 'SVHN_multi_crop_normalized_64.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
