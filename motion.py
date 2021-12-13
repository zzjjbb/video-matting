from skimage.registration import phase_cross_correlation
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import ransac
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform, PolynomialTransform, warp


def find_transform(image1, image2, matte1, matte2):
    block_size = 120
    image_size = image1.shape
    # slices_list = [[(slice(i, i + block_size), slice(j, j + block_size))
    #                 for j in range(0, image_size[1], block_size)]
    #                for i in range(0, image_size[0], block_size)]
    key_points = []
    motion = []
    for i in range(0, image_size[0], block_size):
        for j in range(0, image_size[1], block_size):
            s = (slice(i, i + block_size), slice(j, j + block_size))
            m1, m2 = matte1[s], matte2[s]
            if np.sum(m1 * m2) > block_size ** 2 * 0.8:
                i1, i2 = image1[s], image2[s]
                motion.append(phase_cross_correlation(i1, i2, reference_mask=m1, moving_mask=m2)[::-1])
                key_points.append(np.array([j + block_size / 2, i + block_size / 2]))
    key_points = np.stack(key_points)
    data = [key_points, key_points + motion]
    min_points = round(0.5 * len(key_points))
    return ransac(data, ProjectiveTransform, min_points, 1)[0]


def quiver_plot(image1, image2, matte1, matte2):
    block_size = 120
    image_size = image1.shape
    # slices_list = [[(slice(i, i + block_size), slice(j, j + block_size))
    #                 for j in range(0, image_size[1], block_size)]
    #                for i in range(0, image_size[0], block_size)]
    motion = []
    for i in range(0, image_size[0], block_size):
        motion_row = []
        for j in range(0, image_size[1], block_size):
            s = (slice(i, i + block_size), slice(j, j + block_size))
            m1, m2 = matte1[s], matte2[s]
            if np.sum(m1 * m2) > block_size ** 2 * 0.8:
                i1, i2 = image1[s], image2[s]
                motion_row.append(phase_cross_correlation(i1, i2, reference_mask=m1, moving_mask=m2)[::-1])
            else:
                motion_row.append((0, 0))
        motion.append(motion_row)

    motion = np.array(motion)
    plt.quiver(-motion[:, :, 0], motion[:, :, 1])
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    i1 = rgb2gray(imread("raw/027.png"))
    i2 = rgb2gray(imread("raw/028.png"))
    m1 = rgb2gray(imread("matte/027.png"))
    m2 = rgb2gray(imread("matte/028.png"))
    quiver_plot(i1, i2, 1 - m1, 1 - m2)
    t = find_transform(i1, i2, 1 - m1, 1 - m2)
    iw = warp(i1, t)
    plt.imshow(np.abs(i1 - i2) > 0.1)
    plt.show()
    plt.imshow(np.abs(iw - i2) > 0.1)
    plt.show()

# pm = np.array([[(0, 0) if i is None else i for i in j] for j in motion])
# plt.quiver(-pm[:, :, 1], pm[:, :, 0])
# plt.gca().invert_yaxis()
# plt.show()
