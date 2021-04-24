'''
Panorama pictures convert to stereo projection
Usage:
    $ pyhton3 stereo.py <file_path> <d>
    
    file_path(str): the pano pic file
    d(float)      : d in (0..1.]
'''
import argparse
from math import pi, atan, cos, sin, acos, sqrt, tan
import sys
from scipy.interpolate import RectBivariateSpline
import numpy as np
import cv2


def projection_angle(x, d):
    """
    The function is the inversion of eq(1)in W. Yang's m-p YOLO paper.
    In the case of d=1:
    phi = atan(4*xp/(4-xp**2)) if xp != xp_max else np.pi/2
    theta = atan(4*yp/(4-yp**2)) if yp != yp_max else np.pi/2
    Args:
        x: the symbol xp or yp in eq(1) of W. Yang's m-p YOLO paper.
        d: the de-center from the center of a sphere in W. Yang's m-p YOLO paper.
    Return:
        project_angle: theta or phi in eq(1) of W. Yang's m-p YOLO paper.
    """
    x_max = (1 + d) / d
    numerator = -2 * d * x ** 2 + 2 * (d + 1) * sqrt((1 - d ** 2) * x ** 2 + (d + 1) ** 2)
    denominator = 2 * (x ** 2 + (d + 1) ** 2)
    if 0 < x < x_max:
        project_angle = acos(numerator / denominator)
    elif x < 0:
        project_angle = - acos(numerator / denominator)
    elif x == x_max:
        project_angle = pi/2.
    else:
        raise Exception('invalid input args')
    return project_angle

def pano2stereo(pic, distance=1.):
    '''
    The main function for panorama picture transfrom to stereo projection,
    and save the transformed pic as 'face_0.jpg', 'face_1.jpg', 'face_2.jpg', 'face_3.jpg'.
    Each pic represents different projection face.
    Args:
        pic: input panorama picture
    '''
    frames = []
    input_img = pic
    height, width, _ = input_img.shape
    d = distance
    xp_max = (1 + d) / d  # in the case of d=1, it is 2
    yp_max = (1 + d) / d  # in the case of d=1, it is 2
    xp_domain = xp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    yp_domain = yp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    delta_rad = 2 * pi / width  # get the rads of each pixel

    for face in range(4):
        print('generating face', face)
        output_img = np.zeros((height, height, 3))
        # interpolate function for each channel which is provided by scipy
        interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 0])
        interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 1])
        interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 2])
        pano_x = np.zeros((height, 1))
        pano_y = np.zeros((height, 1))

        # longitude (phi) and latitude (theta) is the angular information from center of the sphere
        for j, xp in enumerate(xp_domain):
            phi = projection_angle(xp, d)
            pano_x[j] = (width / 2.0 + (phi / delta_rad))

        for i, yp in enumerate(yp_domain):
            theta = projection_angle(yp, d)
            pano_y[i] = height/2.0 + (theta/delta_rad)

        output_img[:, :, 0] = interpolate_0(pano_y, pano_x)
        output_img[:, :, 1] = interpolate_1(pano_y, pano_x)
        output_img[:, :, 2] = interpolate_2(pano_y, pano_x)

        cv2.imwrite('face_'+str(face)+'_'+str(d)+'.jpg', output_img)
        frames.append(output_img)
        # change the projection face for the origin panorama
        input_img = np.concatenate(
            (input_img[:, int(width/4):, :], input_img[:, :int(width/4), :]), axis=1)
    return frames

def stereo2pano(in_pic):
    '''
    Stereo Projection picture to transform back to panorama
    Args:
        in_pic: the stereo pic you want to transform
    
    Return:
        output_img(np.array): the pano image
    '''
    input_img = in_pic
    d = 1.
    height = input_img.shape[0]
    width = input_img.shape[1]
    output_img = np.zeros((height, height, 3))

    xp_domain = np.arange(-pi/2., pi/2., pi/height) + pi/height
    yp_domain = np.arange(-pi/2., pi/2., pi/height) + pi/height

    # interpolate function for each channel which is provided by scipy
    interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 0])
    interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 1])
    interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), input_img[:, :, 2])

    for j, phi in enumerate(xp_domain):
        for i, theta in enumerate(yp_domain):

            stereo_x = 2*sin(phi)/(1+cos(phi)) * width/4 + width/2
            stereo_y = 2*sin(theta)/(1+cos(theta)) * height/4 + height/2

            output_img[i, j, 0] = interpolate_0([stereo_y], [stereo_x])
            output_img[i, j, 1] = interpolate_1([stereo_y], [stereo_x])
            output_img[i, j, 2] = interpolate_2([stereo_y], [stereo_x])

    return output_img

def realign_bbox(center_x, center_y, width, height, face):
    if face == 3:
        face = -1
    def safe_atan(x):
        if x == 2: return pi/2
        elif x == -2: return -pi/2
        else: return atan(4*x/(4-x**2))

    xp = 4*(center_x-0.5)
    phi = safe_atan(xp)
    phi = phi + face*pi/2
    if phi > 2*pi:
        phi = phi-4*pi
    center_phi = phi/(2*pi)+0.5
    
    yp = 4*(center_y-0.5)
    theta = safe_atan(yp)
    center_theta = theta/pi+0.5
    
    def realign_border(center, line):
        vertex_1 = 4*(center-0.5-line/2)
        vertex_2 = 4*(center-0.5+line/2)
        angle_1 = safe_atan(vertex_1)
        angle_2 = safe_atan(vertex_2)
        return np.absolute(angle_2-angle_1)

    pano_width = realign_border(center_x, width)/(2*pi)
    pano_height = realign_border(center_y, height)/pi

    return center_phi, center_theta, pano_width, pano_height

def merge_stereo(stereos):
    print('Merging the projected pictures back...')
    frame_0 = stereo2pano(stereos[0]) # from -pi/2 ~pi/2
    print('====First  Picture====')
    frame_1 = stereo2pano(stereos[1]) # from 0 ~ pi
    print('====Second Picture====')
    frame_2 = stereo2pano(stereos[2]) # from pi/2 ~ pi and -pi ~ -pi/2
    print('====Third  Picture====')
    frame_3 = stereo2pano(stereos[3]) # from -pi/2 ~ 0
    print('====Forth  Picture====')
    stride = int(frame_2.shape[1]/2)
    
    pano_1 = np.concatenate([frame_3, frame_1], axis=1)
    pano_2 = np.concatenate([frame_2[:, stride:, :], frame_0, frame_2[:, :stride, :]], axis=1)

    cv2.imwrite('./merge_pano.jpg', (pano_1 + pano_2)/2)

    return (pano_1 + pano_2)/2

def main():
    '''
    just for testing...
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--p2s', help='Path to panorama file.')
    parser.add_argument('--d', help='Postion of Projection', default=1., type=float)
    parser.add_argument('--s2p', help='Path to stereo file.')
    parser.add_argument('--output', help='Path to output file') # ToDo(kevin): set a default
    args = parser.parse_args()

    if (args.p2s):
        pano = cv2.imread(args.p2s)
        pano2stereo(pano, args.d)
    if (args.s2p):
        stereo = cv2.imread(args.s2p)
        cv2.imwrite(args.output, stereo2pano(stereo))

if __name__ == '__main__':
    main()
