# Einlesen von Bildern mit Matplotlib , cv2 ,
import os

import matplotlib.pyplot as plt

import cv2  # install: opencv-python
from PIL import Image  # install pillow
from bs4 import element
import numpy as np


image_path = 'pics/RT.png'
img_to_save = 'pics/RT_NEW.png'
img_to_save2 = 'pics/RT_Inverted.png'
# image = plt.imread(image_path)
image = cv2.cvtColor (cv2.imread (image_path), cv2.COLOR_BGR2RGB)
# image = Image.open(os.path.join(image_path))

# Ausgeben von Bildern:

nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def name_of_func(a, the_list):
    new_List = []
    for y in the_list:
        new_List.append (y * a)
    return new_List


def listBroadcast2(a, listname):
    for idx in range (len (listname)):
        print (idx, element)


def show_image(image_to_show):
    plt.imshow (image_to_show)
    plt.show ()


def find_extrema(image_for_extrema):
    gray_value_matrix = image_for_extrema.reshape(image_for_extrema.shape[0] * image_for_extrema.shape[1], image.shape[2])
    flat_values = gray_value_matrix.T
    flat_values = flat_values[0:3]
    flat_values = flat_values.T
    max_elem = np.amax (flat_values)
    min_elem = np.amin (flat_values)
    print ("max_elem: ", max_elem)
    print ("min: ", min_elem)
    return max_elem, min_elem


def print_gray_histgram(image_for_histo):
    gray_value_matrix = image_for_histo.reshape(image_for_histo.shape[0] * image_for_histo.shape[1], image.shape[2])
    flat_values = gray_value_matrix.T
    flat_values = flat_values[0:3]
    flat_values = flat_values.T
    print (flat_values)
    print (flat_values.shape)
    print (flat_values.ndim)
    max_elem, min_elem = find_extrema(image_for_histo)
    plt.title ("x-Ray")
    plt.xlabel ("Wert")
    plt.ylabel ("Häufigkeit")
    n, bins, patches = plt.hist (flat_values, bins=256, range=[0, 256])
    scalar = 256 / max_elem
    plt.show()
    return scalar

def invert_gray_scale(image):
    gray_value_matrix = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    flat_values = gray_value_matrix.T
    flat_values = flat_values[0:3]
    imageMatrix = flat_values.T
    counter = 0
    for elem in imageMatrix:
        for el in elem:
            #for e in el:
                new_elem = 256 - el
                imageMatrix[counter][:] = new_elem
        counter = counter+1
    newImage = imageMatrix.reshape(image.shape[0] , image.shape[1], image.shape[2])
    return newImage


if __name__ == '__main__':
    # gaussian_numbers = np.random.normal (size=10000)
    # Matrix1 = np.arange (12).reshape (4, 3)

   #image = [[[33, 33, 33 ]]]
    #invert_gray_scale(image)
    #show_image(image)

    #gray_value_matrix = matrix.reshape (matrix.shape[0] * matrix.shape[1], matrix.shape[2])
    #print ("len(gray_value_matrix) ", len (gray_value_matrix))
    #flat_values = gray_value_matrix[:][:]

    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)
    #flat_values = gray_value_matrix.T
    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)
    #flat_values = flat_values[:-1]

    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)


    # np.amax(flat_values, axis=None, out=None, keepdims=<no value>, initial=<no value>)


    # for elem in flat_values:

    #n, bins, patches = plt.hist (flat_values, bins=100, range=[0, 1])
    # print("n: ", n, sum(n))
    # print("bins, ", bins)
    #
    # print("patches ", patches)
    # plt.show()
    # print(name_of_func(2, nested_list))
    # print(name_of_func(2, nested_list))

    # print ("scalar ", scalar)
    # print ("scalar*max_elem ", scalar * max_elem)
    # print ('type: ', type (image))
    # print ('data-type: ', image.dtype)
    # print ('Dimensions: ', image.ndim)
    # print ('shape: ', image.shape)
    # print('matrix[1][1][:3]', image[1][1][:3])
    # for elem in image[:][:]:
    #     image[:][1][:3] =  scalar * image[1][1][:3]
    scalar = print_gray_histgram(image)

    image2 = image[:][:][:] * scalar
    print_gray_histgram(image2)
    print ("--------------------------")
    print ('type: ', type (image))
    print ('data-type: ', image.dtype)
    print ('Dimensions: ', image.ndim)
    print ('shape: ', image.shape)
    print ('matrix[1][1][:3]\n', image)



    import cv2
    import matplotlib.pyplot as plt

    from PIL import Image
    picture = Image.open(image_path)

    # Get the size of the image

    cv2.imshow('matrix', image2)
    cv2.waitKey(0)
    cv2.imwrite(img_to_save, image2)

    inverted_image = invert_gray_scale(image2)
    show_image(inverted_image)
    print(inverted_image)
    cv2.imshow('matrix', inverted_image)
    cv2.waitKey(0)
    cv2.imwrite(img_to_save2, image2)


