"""
@Time: 2024/9/2 22:07
@Author: xujinlingbj
@File: image_utils.py
"""
from collections import defaultdict

from pylatexenc.latex2text import LatexNodes2Text
from PIL import Image

from PIL import ImageFile

from data_process.utils import load_json_file, replace_multiple_spaces

ImageFile.LOAD_TRUNCATED_IMAGES = True


import cairosvg
import io


def open_svg(svg_path):
    """将SVG文件转换为Pillow图像对象"""
    # 读取SVG文件
    with open(svg_path, 'rb') as svg_file:
        svg_data = svg_file.read()

    # 将SVG数据转换为PNG
    png_data = cairosvg.svg2png(bytestring=svg_data)

    # 使用Pillow打开PNG数据
    image = Image.open(io.BytesIO(png_data))
    if is_transparent(image):
        image = convert_transparent_to_white_background(image)
    return image


def is_transparent(image):

    image = image.convert('RGBA')
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = image.getpixel((x, y))
            if a < 255:
                return True
    return False


def convert_transparent_to_white_background(image):
    image = image.convert('RGBA')
    transparent_pixels_found = False
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = image.getpixel((x, y))
            if a < 255:
                transparent_pixels_found = True
                break
        if transparent_pixels_found:
            break

    if transparent_pixels_found:
        white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        white_background.paste(image, mask=image)
        return white_background
    else:
        return image


def group_raw_test_data(path):
    test_data = load_json_file(path)
    group_data = defaultdict(list)
    for category in test_data:
        keyword = category["keyword"].lower()
        example = category["example"]
        for i in range(len(example)):
            question = example[i]['question']
            question = LatexNodes2Text().latex_to_text(question).replace('\n', '')
            question = replace_multiple_spaces(question)
            example[i]['question'] = question
        group_data[keyword] = example
    return group_data
