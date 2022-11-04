
# Einlesen von Bildern mit Matplotlib , cv2 ,
import os

import matplotlib.pyplot as plt
import cv2 # install: opencv-python
from PIL import Image # install pillow
from bs4 import element
import numpy as np

image_path = 'pics/RT.png'
img_to_save = 'pics/RT_NEW.png'
image = plt.imread(image_path)
#image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#image = Image.open(os.path.join(image_path))

# Ausgeben von Bildern:

nested_list = [[1,2,3],[4,5,6],[7,8,9]]


def name_of_func(a , the_list):
    new_List = []
    for y in the_list:
       new_List.append(y * a)
    return new_List

def listBroadcast2(a, listname):
    for idx in range(len(listname)):
        print (idx, element)



if __name__ == '__main__':
    #plt.imshow(image)



    gaussian_numbers = np.random.normal(size=10000)
    Matrix1 = np.arange(12).reshape(4,3)

    plt.imshow(image)
    plt.show()
    matrix = image[...]
    #print (matrix)
    print('type: ', type(matrix))
    print('data-type: ', matrix.dtype)
    print('Dimensions: ', matrix.ndim)
    print('shape: ', matrix.shape)

    print('len(matrix[1][1][:]) ', len(matrix[1][1][:]))
    print('matrix[1][1][:3]', matrix[1][1][:3])
    gray_value_matrix = matrix.reshape(matrix.shape[0] * matrix.shape[1] , matrix.shape[2])
    print( "len(gray_value_matrix) ", len(gray_value_matrix))
    flat_values = gray_value_matrix[:][:]
    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)
    flat_values = gray_value_matrix.T
    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)
    flat_values = flat_values[:-1]

    # print(flat_values)
    # print(flat_values.shape)
    # print(flat_values.ndim)
    flat_values = flat_values.T
    print(flat_values)
    print(flat_values.shape)
    print(flat_values.ndim)

    # np.amax(flat_values, axis=None, out=None, keepdims=<no value>, initial=<no value>)
    max_elem = np.amax(flat_values)
    min_elem = np.amin(flat_values)
    print("max_elem: ", max_elem)
    print("min: ", min_elem)
    plt.title("x-Ray")
    plt.xlabel("Wert")
    plt.ylabel("Häufigkeit")

    #for elem in flat_values:

    n, bins, patches = plt.hist(flat_values, bins = 100, range = [0 , 1])
    # print("n: ", n, sum(n))
    # print("bins, ", bins)
    #
    # print("patches ", patches)
    plt.show()
    # print(name_of_func(2, nested_list))
    # print(name_of_func(2, nested_list))
    scalar = 1 /  max_elem
    print("scalar ", scalar)
    print("scalar*max_elem ", scalar*max_elem)
    print('type: ', type(image))
    print('data-type: ', image.dtype)
    print('Dimensions: ', image.ndim)
    print('shape: ', image.shape)
    #print('matrix[1][1][:3]', image[1][1][:3])
    # for elem in image[:][:]:
    #     image[:][1][:3] =  scalar * image[1][1][:3]
    image = image[:][:][:] * scalar
    print("--------------------------")
    print('type: ', type(image))
    print('data-type: ', image.dtype)
    print('Dimensions: ', image.ndim)
    print('shape: ', image.shape)
    print('matrix[1][1][:3]',  image)
    print(image)
    im = Image.fromarray(image)
    import scipy.misc
    scipy.misc.imsave(img_to_save, image)
