#Yanying Wu_Morgan Lab

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import circle_fit
import skimage.exposure

image_folder = 'Testing_Image/'

image_file_list = os.listdir(image_folder)

if '.DS_Store' in image_file_list:
    image_file_list.remove('.DS_Store')

for image_file in image_file_list:

    vis_image = cv2.imread(image_folder + image_file)

    gray_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)

    gray_image = skimage.exposure.adjust_gamma(gray_image, 0.5)

    threshold_image = cv2.adaptiveThreshold(gray_image, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 99, 2)

    num_labels, label_image, stats, centroids = cv2.connectedComponentsWithStats(threshold_image, 4, cv2.CV_32S)

    mask = (stats[:, 2] > gray_image.shape[1] / 2) & (stats[:, 3] > gray_image.shape[0] / 2) & (stats[:, 0] != 0) & (stats[:, 1] != 0) & (stats[:, 0] + stats[:, 2] != gray_image.shape[1]) & (stats[:, 1] + stats[:, 3] != gray_image.shape[0])
    mask[0] = False

    ring_image = np.zeros(gray_image.shape, np.uint8)
    for label in np.nonzero(mask)[0]:
        ring_image[label_image == label] = 255

    ring_image = cv2.morphologyEx(ring_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    contours, hierarchy = cv2.findContours(ring_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    plt.imshow(cv2.drawContours(vis_image.copy(), contours, -1, (0, 255, 0), 2))
    plt.show()

    # plt.imshow(cv2.drawContours(vis_image.copy(), contours, len(contours) - 1, (0, 255, 0), 2))
    # plt.show()

    innor_ring = np.squeeze(contours[-1])

    x_center, y_center, inner_radius, _ = circle_fit.least_squares_circle(innor_ring)

    def get_rint(num):
        return np.rint(num).astype(int)

    cv2.circle(vis_image, (get_rint(x_center), get_rint(y_center)), get_rint(inner_radius), (0, 255, 0), 3)

    outer_radius = np.sqrt((np.count_nonzero(ring_image) + np.pi * inner_radius**2) / np.pi)

    cv2.circle(vis_image, (get_rint(x_center), get_rint(y_center)), get_rint(outer_radius), (0, 255, 0), 3)

    plt.title(image_file+': '+str((outer_radius - inner_radius) / 0.2093))
    plt.imshow(vis_image)
    plt.show()
