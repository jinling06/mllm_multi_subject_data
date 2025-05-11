"""
@Time: 2024/8/26 20:08
@Author: xujinlingbj
@File: make_final_train_data.py
"""
import sys

from data_process.utils import load_json_file, save_json_file, merge_multi_file


if __name__ == '__main__':
    file_list = sys.argv[1]
    save_path = sys.argv[2]
    if not isinstance(file_list, list):
        file_list = file_list.split(' ')
    print(f'file_list={file_list}')
    merge_multi_file(file_list=file_list, save_path=save_path)
