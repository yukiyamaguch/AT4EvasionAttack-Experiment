import cv2
from utils import l2_norm_of_perturbation

img = cv2.imread('dataset/victim/10.png') / 255.
img = cv2.resize(img, (48,48))
#gx10 = cv2.imread('output/AdvGAN/test/0010-10.png') / 255.
#gx30 = cv2.imread('output/AdvGAN/test/0030-10.png') / 255.
#gx50 = cv2.imread('output/AdvGAN/test/0050-10.png') / 255.
#gx70 = cv2.imread('output/AdvGAN/test/0070-10.png') / 255.

for i in range(60):
    n = i + 1
    gx = cv2.imread(f'output/AdvGAN/test/0{n:02}0-10.png') / 255.
    print(f'img-gx{n}0 norm: {l2_norm_of_perturbation(gx-img)}')

#print(f'img-gx10 norm: {l2_norm_of_perturbation(gx10-img)}')
#print(f'img-gx30 norm: {l2_norm_of_perturbation(gx30-img)}')
#print(f'img-gx50 norm: {l2_norm_of_perturbation(gx50-img)}')
#print(f'img-gx70 norm: {l2_norm_of_perturbation(gx70-img)}')
