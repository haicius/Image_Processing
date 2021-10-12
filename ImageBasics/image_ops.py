# Author: 海子
# Create Time: 2021/8/16
# FileName: image_ops
# Descriptions:使用numpy, opencv， matplotlib,图像，镜像操作
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageOpsFromScratch(object):
    def __init__(self, image_file):
        self.image_file = image_file

    def read_this(self, gray_scale=False):
        image_src = cv2.imread(self.image_file)
        if gray_scale:
            image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        else:
            image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        return image_rgb

    def mirror_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_mirror = np.fliplr(image_rgb)
        if with_plot:
            self.plot_it(orig_matrix=image_rgb, trans_matrix=image_mirror, head_text='Mirrored', gray_scale=gray_scale)
            return None
        return image_mirror

    def flip_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_flip = np.flipud(image_rgb)
        if with_plot:
            self.plot_it(orig_matrix=image_rgb, trans_matrix=image_flip, head_text='Flipped', gray_scale=gray_scale)
            return None
        return image_flip

    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text(head_text)
        if not gray_scale:
            ax1.imshow(orig_matrix)
            ax2.imshow(trans_matrix)
            plt.show()
        else:
            ax1.imshow(orig_matrix, cmap='gray')
            ax2.imshow(trans_matrix, cmap='gray')
            plt.show()
        return True


imo = ImageOpsFromScratch(image_file='lena.png')

# Mirroring
imo.mirror_this()
imo.mirror_this(gray_scale=True)

# Flipping
imo.flip_this()
imo.flip_this(gray_scale=True)
