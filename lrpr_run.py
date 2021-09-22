#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:21:32 2021

@author: soominkwon
"""

import numpy as np
from lrpr_via_cgls import lrpr_fit
import matplotlib.pyplot as plt
from generate_lrpr import generateLRPRMeasurements
import cv2
from PIL import Image

image_name = 'image_tensor_small.npz'
with np.load(image_name) as data:
    tensor = data['arr_0']
n1, n2, q_dim = tensor.shape

L = 3 # number of modulations
m_dim = n1 * n2 * L

images, Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim, L=L)

U, B = lrpr_fit(Y=Y, A=A, rank=1)

X_hat = U @ B.conj().T
vec_first_image = X_hat[:, 0]

first_image = np.reshape(
    vec_first_image, (images.shape[0], images.shape[1]), order='F')

# Currently we only have three images, i.e., q = fps * duration = 3
fps = 1
duration = 3

# Define the videowriter, the format of the reconstructed video is *.avi.
out = cv2.VideoWriter(
    'reconstructed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (n2, n1), False)

for k in range(np.int(fps * duration)):
    # Fetching the kth frame from the reconstructed array.
    x_k = np.abs(np.reshape((X_hat[:, k]), (n1, n2), 'F'))
    # Normalize the frame.
    frame = 255.0 * x_k / np.ptp(x_k)
    # Write the constructed frame to the video file
    out.write(frame.astype(np.uint8))

# Release the reconstructed video.
out.release()

plt.imshow(np.abs(images[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(first_image), cmap='gray')
plt.title('Reconstructed Image via LRPR2')
plt.show()   
