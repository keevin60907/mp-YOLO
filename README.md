# Multi Projection YOLO
The reproduce result for 'Object Detection in Equirectangular Panorama'

# Usage
$ python3 detection <pano_picture> <result_picture>

pano_picture is the file path of the panorama picture you want to detect.

result_picture is the output path for visualizin the result.

# Impelement Detail
The program is based on 'Object Detection in Equirectangular Panorama'(2018 ICPR)(https://arxiv.org/abs/1805.08009), and the YOLO model is replaced with YOLOv3. Soft-NMS selection have not been included in current version.

# Environmet
Opencv >= 3.4

# License
This repository is released under the MIT License (refer to the LICENSE file for details).
