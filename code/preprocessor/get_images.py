import sys

sys.path.append("/home/ubuntu/code/fingerprint_python")
import os
from constants import constants as fpconst
import numpy as np
from scipy import misc
from PIL import Image


def convert_to_shape(arr):
    # print(max_height)
    height = arr.shape[0]

    total_pad = fpconst.max_height - height
    # print(total_pad)
    pad_height_up = int(total_pad / 2)
    pad_height_down = total_pad - pad_height_up

    img = np.zeros((fpconst.max_height, fpconst.max_width))
    start = pad_height_up
    end = pad_height_up + height
    # print(start)
    # print(end)

    img[start:end, :] = arr

    return img


def get_bounded_img(img_location):
    print(img_location)
    arr = misc.imread(img_location)
    minx = arr.shape[0]
    miny = arr.shape[1]
    maxx = 0
    maxy = 0

    # img = Image.fromarray(arr)
    # print(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 255:
                minx = min(minx, i)
                maxx = max(maxx, i)
                miny = min(miny, j)
                maxy = max(maxy, j)

    arr = arr[minx:maxx, miny:maxy]

    img = Image.fromarray(arr)

    next_width = fpconst.max_width
    next_height = int(next_width * (img.height / img.width))
    img = img.resize((next_width, next_height), Image.ANTIALIAS)
    arr = np.array(img)
    arr = convert_to_shape(arr)
    return arr


# Returns train_data in the shape of m* (3*region_height * 3 * region_width * 1 )
def extract_from_image(file_name):
    img = get_bounded_img(file_name)
    height = fpconst.region_height_size
    width = fpconst.region_width_size

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    # print(img_rows, img_cols)
    # print(height,width)

    result = []

    start_row = 0
    start_col = 0

    while True:
        # print("Current Row ::", start_row)
        # print("Current Col ::", start_col)

        if start_row >= img_rows:
            break
        data = np.zeros((3 * height, 3 * width))
        ## Extract Data

        row = start_row - height
        col = start_col - width
        count = 1
        r = 0
        c = 0
        while count < 9:
            # print(count)
            # print(r,r+height,c,c+width)
            # print(row,row+height,col,col+width)
            if row >= 0 and (row + height) < img_rows and col >= 0 and (col + width) < img_cols:
                data[r:r + height, c:c + width] = img[row:row + height, col:col + width]
            if count % 3 == 0:
                row = row + height
                col = start_col - width
                c = 0
                r = r + height
            else:
                col = col + width
                c = c + width

            count += 1

        data = data.reshape((3 * height, 3 * width, 1))
        result.append(data)

        # input("Test")

        start_col = start_col + width
        if start_col > img_cols:
            start_row = start_row + height
            start_col = 0

    return result


def extract_from_image_single_region(file_name):
    img = get_bounded_img(file_name)
    height = fpconst.region_height_size
    width = fpconst.region_width_size

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    result = []

    start_row = 0
    start_col = 0

    print(img_rows, img_cols)

    while True:
        if start_row >= img_rows:
            break

        end_row = start_row + height
        end_col = start_col + width

        data = img[start_row:end_row, start_col:end_col]
        data = data.reshape((height, width, 1))
        result.append(data)

        start_col = start_col + width
        if start_col >= img_cols:
            start_row = start_row + height
            start_col = 0

    return result


def extract_all_regions(loc):
    train_data = []
    for file in os.listdir(loc):
        file_name = loc + "/" + file
        data = extract_from_image_single_region(file_name)
        print("Image Extracted")
        for i in range(len(data)):
            train_data.append(data[i])
        break #TODO

    print(len(train_data))

    return np.array(train_data)


def get_data_image(loc):
    train_data = []
    for file in os.listdir(loc):
        file_name = loc + "/" + file
        data = extract_from_image(file_name)
        print("Image Extracted")
        for i in range(len(data)):
            train_data.append(data[i])

    print(len(train_data))

    return np.array(train_data)


if __name__ == "__main__":
    pass
