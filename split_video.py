#! /usr/bin/env python
# coding: utf-8

"""
将一段视频的关键帧切出来 供合并图片使用
"""

import cv2
import numpy as np
import os
import sys
import click
import time


class VideoSplitter(object):
    """分离视频关键帧"""

    def __init__(self, video_name):
        self.video_name = os.path.abspath(video_name)
        if not os.path.isfile(self.video_name):
            sys.exit("{} not exists.".format(video_name))
        folder_head, folder_tail = os.path.split(self.video_name)
        video_name_without_ext, _ = os.path.splitext(folder_tail)
        self.dest_folder = os.path.join(folder_head, "{}_video_slices_{}".format(video_name_without_ext, int(time.time() * 1000)))
        try:
            os.mkdir(self.dest_folder)
        except Exception:
            pass

    def next_capture_name(self, preffix, count):
        return os.path.join(self.dest_folder, "{}_{:0>10}.jpg".format(preffix, count))

    def save_image_to_file(self, filename, image):
        cv2.imwrite(filename, image)
        print "Save To: ", filename

    def gogogo(self):
        cap = cv2.VideoCapture(self.video_name)
        image_count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            self.save_image_to_file(self.next_capture_name("cap", image_count), frame)
            # # 如何判断结束了呢
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            image_count += 1

        cap.release()
        cv2.destroyAllWindows()


@click.command()
@click.option('--video_name', default="src_video_9.mp4", help='dest image width')
def main(video_name):
    VideoSplitter(video_name).gogogo()


if __name__ == '__main__':
    main()


