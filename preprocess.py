
import numpy as np
import cv2

from skimage.transform import radon


filename = '/home/ubuntu/ocr/table_structure_recognition/p2.jpg'
# Load file, converting to grayscale
img = cv2.imread(filename)
I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = I.shape
# If the resolution is high, resize the image to reduce processing time.
if (w > 640):
    I = cv2.resize(I, (640, int((h / w) * 640)))
I = I - np.mean(I)  # Demean; make the brightness extend above and below zero
# Do the radon transform
sinogram = radon(I)
# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
rotation = np.argmax(r)
print('Rotation: {:.2f} degrees'.format(90 - rotation))
def rotate_img(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img
# Rotate and save with the original resolution
# M = cv2.getRotationMatrix2D((w/2, h/2), 90 - rotation, 1)
# dst = cv2.warpAffine(img, M, (w, h))

dst = rotate_img(img, -90 + rotation)

# 确认90整数倍
from paddleocr import PaddleOCR
sample_filename = '/home/ubuntu/ocr/table_structure_recognition/p2_rotated.jpg'
ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
angle_cls = ocr_model.ocr(dst, det=False, rec=False, cls=True)
angle_cls # 0 continue

dst1 = rotate_img(dst, int(angle_cls[0][0]))

cv2.imwrite('/home/ubuntu/ocr/table_structure_recognition/p2_rotated.jpg', dst1)