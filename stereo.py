'''
Panorama pictures convert to stereo projection
'''
from math import pi, atan, cos, sin
from scipy.interpolate import RectBivariateSpline
import numpy as np
import cv2

def pano2stereo(pic):
    '''
    The main function for panorama picture transfrom to stereo projection,
    and save the transformed pic as 'face_0.jpg', 'face_1.jpg', 'face_2.jpg', 'face_3.jpg'.
    Each pic represents different projection face.

    Args:
        pic(str): the path of input picture

    '''
    input_img = cv2.imread(pic)
    for face in range(4):
        height = input_img.shape[0]
        width = input_img.shape[1]
        rads = 2*pi / width # get the rads of each pixel
        output_img = np.zeros((height, height, 3))

        img_x = np.arange(-1.0, 1.0, 2.0/height) + 1.0/height
        img_y = np.arange(-1.0, 1.0, 2.0/height) + 1.0/height
        # (pixel_x[i, j], pixel_y[i, j]) means the point[i, j] = (x, y)
        pixel_x, pixel_y = np.meshgrid(img_x, img_y)

        # interpolate function for each channel which is provided by scipy
        interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 0])
        interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 1])
        interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 2])

        for i in range(height):
            for j in range(height):
                # longitude and latitude is the anular information from center of the sphere
                # phi and theta is the angle measure from the position (d = 1)
                longitude = 2*atan(pixel_x[i, j])
                phi = 2*sin(longitude) / (1+cos(longitude))

                latitude = atan(pixel_y[i, j])
                theta = 2*sin(latitude) / (1+cos(latitude))

                pano_x = width/2.0 + (phi/rads)
                pano_y = height/2.0 + (theta/rads)

                output_img[i, j, 0] = interpolate_0([pano_y], [pano_x])
                output_img[i, j, 1] = interpolate_1([pano_y], [pano_x])
                output_img[i, j, 2] = interpolate_2([pano_y], [pano_x])

        cv2.imwrite('face_'+str(face)+'.jpg', output_img)
        # change the projection face for the origin panorama
        input_img = np.concatenate(
            (input_img[:, int(width/4):, :], input_img[:, :int(width/4), :]), axis=1)
def main():
    '''
    just for testing...
    '''
    pano2stereo('example.jpg')

if __name__ == '__main__':
    main()
