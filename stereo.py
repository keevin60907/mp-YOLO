'''
Panorama pictures convert to stereo projection
Usage:
    $ pyhton3 stereo.py <file_path> <d>
    
    file_path(str): the pano pic file
    d(float)      : d in (0..1.]
'''
import sys
from math import pi, atan, cos, sin, acos, sqrt, tan
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

    # ToDo(kevin): d is currently fixed to 1, could you generalize it to any d belongs to [0, 1]
    '''
    input_img = cv2.imread(pic)
    d = float(sys.argv[2])
    for face in range(4):
        print('generating face', face)
        height = input_img.shape[0]
        width = input_img.shape[1]
        delta_rad = 2*pi / width  # get the rads of each pixel
        output_img = np.zeros((height, height, 3))
        xp_max = (1+d) / d  # in the case of d=1, it is 2
        yp_max = (1+d) / d  # in the case of d=1, it is 2
        xp_domain = xp_max*(np.arange(-1., 1., 2./height) + 1.0/height)
        yp_domain = yp_max*(np.arange(-1., 1., 2./height) + 1.0/height)

        # interpolate function for each channel which is provided by scipy
        interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 0])
        interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 1])
        interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 2])

        for j, xp in enumerate(xp_domain):
            for i, yp in enumerate(yp_domain):
                # longitude (phi) and latitude (theta) is the angular information from center of the sphere
                # below formula is the inversion of eq(1) at d=1 in Wenyan Yang's m-p YOLO paper.
                # ToDo(kevin): d is currently fixed to 1, could you generalize it to any d belongs to [0, 1]
                # phi = atan(4*xp/(4-xp**2)) if xp != xp_max else np.pi/2
                # theta = atan(4*yp/(4-yp**2)) if yp != yp_max else np.pi/2
                def projection_angle(x, d):
                    numerator = -2*d*x**2 + 2*(d+1)*sqrt((1-d**2)*x**2+(d+1)**2)
                    denominator = 2*(x**2+(d+1)**2)
                    return acos(numerator/denominator)
                phi = projection_angle(xp, d) if xp != xp_max else pi/2
                if xp < 0:
                    phi = -1*phi
                theta = projection_angle(yp, d) if yp != yp_max else pi/2
                if yp < 0:
                    theta = -1*theta

                pano_x = width/2.0 + (phi/delta_rad)
                pano_y = height/2.0 + (theta/delta_rad)

                output_img[i, j, 0] = interpolate_0([pano_y], [pano_x])
                output_img[i, j, 1] = interpolate_1([pano_y], [pano_x])
                output_img[i, j, 2] = interpolate_2([pano_y], [pano_x])

        cv2.imwrite('face_'+str(face)+'_'+str(d)+'.jpg', output_img)
        # change the projection face for the origin panorama
        input_img = np.concatenate(
            (input_img[:, int(width/4):, :], input_img[:, :int(width/4), :]), axis=1)

def stereo2pano(pic):
    input_img = cv2.imread(pic)
    d = 1.
    height = input_img.shape[0]
    width = input_img.shape[1]
    delta_rad = pi / 2
    output_img = np.zeros((height, height, 3))

    xp_domain = np.arange(-1., 1., 2./height) + 1.0/height
    yp_domain = np.arange(-1., 1., 2./height) + 1.0/height

    # interpolate function for each channel which is provided by scipy
    interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 0])
    interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 1])
    interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 2])

    for j, xp in enumerate(xp_domain):
        for i, yp in enumerate(yp_domain):
            alpha = tan(xp*delta_rad)
            beta = tan(yp*delta_rad)

            if alpha < 0:
                stereo_x = width/2.0 + width/4.0 * (-2/alpha - 2*sqrt(1/alpha**2+1))
            else:
                stereo_x = width/2.0 + width/4.0 * (-2/alpha + 2*sqrt(1/alpha**2+1))
            if beta < 0:
                stereo_y = height/2.0 + height/4.0 * (-2/beta - 2*sqrt(1/beta**2+1))
            else:
                stereo_y = height/2.0 + height/4.0 * (-2/beta + 2*sqrt(1/beta**2+1))

            output_img[i, j, 0] = interpolate_0([stereo_y], [stereo_x])
            output_img[i, j, 1] = interpolate_1([stereo_y], [stereo_x])
            output_img[i, j, 2] = interpolate_2([stereo_y], [stereo_x])

    cv2.imwrite('pano'+'.jpg', output_img)

def main():
    '''
    just for testing...
    '''
    #pano2stereo(sys.argv[1])
    stereo2pano(sys.argv[1])

if __name__ == '__main__':
    main()
