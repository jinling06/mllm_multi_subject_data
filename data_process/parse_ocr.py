"""
@Time: 2024/9/11 10:15
@Author: xujinlingbj
@File: parse_ocr.py
"""
# 不要删除，删除会报错，原因未知
import os
import concurrent.futures
import sys
import logging
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR, paddleocr

from data_process.utils import *

paddleocr.logging.disable(logging.DEBUG)


def parse_ocr_thread(version_files, ocr_processor):
    for sample_dir in tqdm(version_files, total=len(version_files), desc='sample'):
        if not os.path.isdir(sample_dir):
            continue
        for file_name in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, file_name)
            if file_path.endswith('.png') and 'Answer' in file_name:
                ocr_save_path = os.path.join(sample_dir, file_name.split('.')[0] + '.json')

                ocr_res = ocr_processor.get_ocr(file_path)
                save_json_file(ocr_save_path, ocr_res)


def parse_ocr_image(data_dir, start=0, end=0):
    ocr_processor = OCRObject()
    thread_num = 1
    all_files = []
    for subject in os.listdir(data_dir):
        # if subject not in ['MATH_v4']:
        #     continue

        subject_dir = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        version_dirs = os.listdir(subject_dir)
        version_files = [os.path.join(subject_dir, x) for x in version_dirs]
        all_files.extend(version_files)

    print(f'all_files num={len(all_files)}')
    print(all_files[0])
    print(start, end)
    # sys.exit(0)
    all_files = all_files[start:end]
    parse_ocr_thread(all_files, ocr_processor)


class OCRObject(object):
    def __init__(self):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    def get_ocr(self, image_path):
        try:
            result = self.paddle_ocr.ocr(image_path, cls=True)
        except:
            print(f'识别失败：{image_path}')
            result = ''
        return result


if __name__ == '__main__':
    data_dir = 'MLLM_data_3_v5'
    start = sys.argv[1]
    end = sys.argv[2]
    start = int(start)
    end = int(end)
    parse_ocr_image(data_dir, start, end)
    pass
