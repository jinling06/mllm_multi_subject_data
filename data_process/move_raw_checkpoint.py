"""
@Time: 2024/8/27 10:27
@Author: xujinlingbj
@File: move_raw_checkpoint.py
"""
import os
import shutil
import sys


def move_raw_files(raw_model_dir, save_dir):
    print('move_raw_files ...')
    print(f'save_dir={save_dir}')
    files = os.listdir(raw_model_dir)
    for file in files:

        if file.endswith('py') or file.endswith('json') or file.endswith('model'):
            file_path = os.path.join(raw_model_dir, file)
            save_path = os.path.join(save_dir, file)
            print(f'{file_path} copy to {save_path}')
            shutil.copy(file_path, save_path)


if __name__ == '__main__':
    save_dir = sys.argv[1]
    move_raw_files(raw_model_dir='model/moss2-2_5b-chat',
                   save_dir=save_dir)
    pass