#! /usr/bin/env python
# coding: utf-8


import click
import numpy as np
from PIL import Image
import cv2
import os


class DestMode:
    """destination image mode"""
    JustDest = 1
    DestWithBac = 2
    DestAndSrcWithBac = 3


class SaveMode:
    JustDest = 1
    Every = 2


class BacSize:
    Width = 1080
    Height = 720


class ImageMerger(object):
    """merge images to one"""

    def __init__(self):
        self.dest_mode = DestMode.DestAndSrcWithBac

        # 计算背景尺寸
        if self.dest_mode == DestMode.DestAndSrcWithBac:
            self.bac_width = BacSize.Width / 2
            self.bac_height = BacSize.Height / 2
        else:
            self.bac_width = BacSize.Width
            self.bac_height = BacSize.Height

        # dest_image
        self.current_total_rgb_counter = None

        # dest 的某个像素点merge了多少图像，最后分像素点取平均
        # self.merge_counter = None

        # dest_image 的信息
        self.dest_image = None
        self.dest_width = 500
        self.dest_height = 500

        # src_image 下一张将要 merge 的照片
        self.next_src_image = None

        # dest_with_bac_image 将dest_image 缩放到 bac 尺寸并加上背景的
        self.dest_with_bac_image = None

        # 将 next_src 缩放到 bac 尺寸并加上背景的
        self.next_src_with_bac_image = None

        #
        self.dest_and_src_with_bac_image = None

        # 从这里读取图片信息
        self.src_folder = "./src_image/"

        # 写入这里
        self.dest_folder = "./dest_image/"

        # 视频写入这里
        self.dest_video_name = "wocao.avi"

        # 初始化一些变量
        self.init()

    def init(self):
        #
        self.current_total_rgb_counter = np.zeros((self.dest_height, self.dest_width, 4))
        # self.merge_counter = np.zeros((self.dest_height, self.dest_width))

    def next_src_image_name(self):
        return self.next_image(self.src_folder)

    def next_image(self, folder_name):
        for f in os.listdir(folder_name):
            filename = os.path.join(os.path.abspath(folder_name), f)
            if os.path.isfile(filename):
                root, ext = os.path.splitext(filename)
                # TODO 增加图片判断
                if ext == ".jpg":
                    yield filename

    def read_image(self, filename):
        return cv2.imread(filename)

    def scale_image(self, dest_width, dest_height, src_image):
        return scale_image(dest_width, dest_height, src_image)

    def save_image_to_file(self, image, filename):
        cv2.imwrite(filename, image)
        print "Save to: {}".format(filename)

    def image_sequence_to_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # DIVX  X264
        out = cv2.VideoWriter(self.dest_video_name, fourcc, 3, (BacSize.Width, BacSize.Height))
        for image_name in self.next_image(self.dest_folder):
            frame = self.read_image(image_name)
            out.write(frame)
            cv2.imshow("frame", frame)
        out.release()
        cv2.destroyAllWindows()

    def next_image_name(self, preffix, count):
        return os.path.join(
            os.path.abspath(self.dest_folder),
            "{}_{:0>10}.jpg".format(preffix, count)
        )

    def add_to_total_pixel_counter(self, src_scaled_image):
        # 首先确定 height width 的范围
        src_height, src_width = src_scaled_image.shape[:2]

        width_padding, height_padding = map(
            lambda (x, y): int((x - y) / 2),
            ((self.dest_width, src_width), (self.dest_height, src_height))
        )

        def transfer_h_w(h, w):
            return h + height_padding, w + width_padding

        # 加和
        for src_h_index in xrange(src_height):
            for src_w_index in xrange(src_width):
                dest_h, dest_w = transfer_h_w(src_h_index, src_w_index)
                # rgb + new_rgb
                ever_total_rgb_with_counter = self.current_total_rgb_counter[dest_h, dest_w].tolist()
                src_rgb = src_scaled_image[src_h_index, src_w_index].tolist()
                src_rgb.append(1)
                for index in xrange(len(ever_total_rgb_with_counter)):
                    ever_total_rgb_with_counter[index] += src_rgb[index]
                self.current_total_rgb_counter[dest_h, dest_w] = ever_total_rgb_with_counter

    def generate_dest_image(self):
        """通过 total_counter 计算出来当前 dest_image"""
        self.dest_image = np.zeros((self.dest_height, self.dest_width, 3))
        for h_index in xrange(self.dest_height):
            for w_index in xrange(self.dest_width):
                b, g, r, c = self.current_total_rgb_counter[h_index, w_index]
                if c != 0:
                    b, g, r = map(
                        lambda x: int(x/c),
                        [b, g, r]
                    )
                self.dest_image[h_index, w_index] = (b, g, r)

    def scale_and_padding(self, total_width, total_height, src_image):
        """将src_image 缩放到合适的尺寸，在周围加上黑边"""
        scaled_image = self.scale_image(total_width, total_height, src_image)
        total_image = np.zeros((total_height, total_width, 3))

        scaled_height, scaled_width = src_image.shape[:2]
        width_padding, height_padding = map(
            lambda (x, y): int((x - y) / 2),
            ((total_width, scaled_width), (total_height, scaled_height))
        )

        def transfer_h_w(h, w):
            return height_padding + h, width_padding + w

        for h_index in xrange(scaled_height):
            for w_index in xrange(scaled_width):
                total_image[transfer_h_w(h_index, w_index)] = scaled_image[h_index, w_index]

        return total_image

    def attach_two_images_left_and_right(self, left_image, right_image):
        total_image = np.zeros((BacSize.Height, BacSize.Width, 3))
        for h_index in xrange(BacSize.Height):
            for w_index in xrange(BacSize.Width):
                # TODO 这里有个坑，total width 为奇数时，最后还得执行 resize 操作。暂且不考虑
                if w_index >= BacSize.Width / 2:
                    pixel = right_image[h_index, w_index - BacSize.Width / 2]
                else:
                    pixel = left_image[h_index, w_index]
                total_image[h_index, w_index] = pixel

        return total_image

    def gogogo(self):

        # 初始化
        image_count = 0
        for image_count, filename in enumerate(self.next_src_image_name(), start=1):
            self.next_src_image = self.read_image(filename)

            # 按照 dest 尺寸缩放好的image
            next_src_scaled_image = self.scale_image(self.dest_width, self.dest_height, self.next_src_image)

            # TODO 分情况 加黑边，还得显示下一张图片 在这里进行
            self.generate_dest_image()

            saving_image = None
            if self.dest_mode == DestMode.JustDest:
                saving_image = self.dest_image
            elif self.dest_mode == DestMode.DestWithBac:
                saving_image = self.scale_and_padding(BacSize.Width, BacSize.Height, self.dest_image)
            else:
                # 多了左右两幅图片加一起的操作
                left_image = self.scale_and_padding(BacSize.Width / 2, BacSize.Height, self.dest_image)
                right_image = self.scale_and_padding(BacSize.Width / 2, BacSize.Height, next_src_scaled_image)
                # 将两幅图片搞在一起
                saving_image = self.attach_two_images_left_and_right(left_image, right_image)

            self.save_image_to_file(saving_image, self.next_image_name("dest", image_count))

            # 将像素信息加到 total_counter 里边
            self.add_to_total_pixel_counter(next_src_scaled_image)
            # self.save_image_to_file(next_src_scaled_image, self.next_image_name(image_count))

        # 生成只含有 dest_image 的图片
        self.generate_dest_image()
        if DestMode.JustDest == self.dest_mode:
            just_dest_image = self.dest_image
        else:
            just_dest_image = self.scale_and_padding(BacSize.Width, BacSize.Height, self.dest_image)

        # save lawst image
        self.save_image_to_file(just_dest_image, self.next_image_name("dest", image_count+1))

        # 生成视频
        self.image_sequence_to_video()


def scale_image(dest_width, dest_height, src_image):
    """首先确定缩放的具体尺寸
    如果src尺寸width, height 任一尺寸均小于dest 则不变化
    """
    # 确定缩放的具体尺寸
    src_height, src_width = src_image.shape[:2]

    if src_height <= dest_height and src_width <= dest_width:
        return src_image

    src_height_scaled, src_width_scaled = 0, 0
    src_rate = float(src_width) / src_height
    dest_rate = float(dest_width) / dest_height
    if src_rate >= dest_rate:  # 照着宽度缩放
        src_width_scaled = dest_width
        src_height_scaled = int(float(src_width_scaled) / src_rate)
    else:
        src_height_scaled = dest_height
        src_width_scaled = int(float(src_height_scaled) * src_rate)

    # 将src的尺寸，设置为scaled的尺寸
    return cv2.resize(src_image, (src_width_scaled, src_height_scaled), interpolation=cv2.INTER_LINEAR)


@click.command()
def main():
    ImageMerger().gogogo()


if __name__ == '__main__':
    main()


"""
cv2图像基本操作
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#basic-ops

几何变换
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

视频简单操作
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
"""