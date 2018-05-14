import sys
sys.path.append("/home/ubuntu/code/fingerprint_python")
import os
from constants import constants
import numpy as np
from scipy import misc
from PIL import Image



def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def convert_to_shape(arr):
    # print(max_height)
    height = arr.shape[0]

    total_pad = constants.max_height - height
    # print(total_pad)
    pad_height_up = int(total_pad / 2)
    pad_height_down = total_pad - pad_height_up

    img = np.zeros((constants.max_height, constants.max_width))
    start = pad_height_up
    end = pad_height_up + height
    # print(start)
    # print(end)

    img[start:end, :] = arr

    return img


def file_name_separator(file_name):
    filename, file_extension = os.path.splitext(file_name)
    print(filename)
    print(file_extension)
    return filename, file_extension


def convert_to_regions(arr):
    start_x = 0
    start_y = 0

    height = arr.shape[0]
    width = arr.shape[1]

    # print(diff_x)
    # print(diff_y)

    result = np.zeros((constants.regions, constants.region_height_size, constants.region_width_size))

    region = 0
    while region < constants.regions:
        if start_y >= width:
            start_x = start_x + constants.region_height_size
            start_y = 0

        result[region, :, :] = arr[
                               start_x:start_x + constants.region_height_size,
                               start_y:start_y + constants.region_width_size]
        region += 1
        start_y = start_y + constants.region_width_size

    # print(result)
    return result


def create_patch(image_location, region_location):
    try:
        os.makedirs(region_location)
    except Exception as e:
        pass
    file_name = os.path.basename(image_location)
    arr = misc.imread(image_location)
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

    next_width = constants.max_width
    next_height = int(next_width * (img.height / img.width))
    img = img.resize((next_width, next_height), Image.ANTIALIAS)
    # img.show()
    arr = np.array(img)
    arr = convert_to_shape(arr)

    img = Image.fromarray(arr)
    # img.show()

    regions = convert_to_regions(arr)

    for region in range(regions.shape[0]):
        data = regions[region]
        name, extension = file_name_separator(file_name)
        save_file = region_location + "/" + name + "_region_" + str(region)
        # if extension is not None and len(extension) > 0:
        #     save_file = save_file + extension
        print(save_file)
        np.save(save_file, data)

    return img.height


if __name__ == "__main__":

    # "/Users/shankaragarwal/Downloads/sd27_data/DB2_B/",
    # "/Users/shankaragarwal/Downloads/sd27_data/DB3_B/",
    # "/Users/shankaragarwal/Downloads/sd27_data/DB4_B/"]

    for i in range(len(constants.locations)):
        location = constants.locations[i]
        save_location = constants.save_locations[i]
        maximum_height = 0
        for file in files(location):
            maximum_height = max(maximum_height, create_patch(location + "/" + file, save_location))
        print(location + "::" + str(maximum_height))
    # create_patch(location + "/102_6.tif")
