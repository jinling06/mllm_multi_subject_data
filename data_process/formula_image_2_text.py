"""
@Time: 2024/10/10 14:40
@Author: xujinlingbj
@File: formula_image_2_text.py
"""
import argparse
import json
import time
from tqdm import tqdm
import evaluate
import random
import re
import unimernet.tasks as tasks

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tabulate import tabulate
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unimernet.common.config import Config

# imports modules for registration
from unimernet.datasets.builders import *
from unimernet.models import *
from unimernet.processors import *
from unimernet.tasks import *
from unimernet.processors import load_processor
"""
pip install --upgrade unimernet
"""

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(raw_image)
        return image


def load_data(image_path, math_file):
    """
    Load a list of image paths and their corresponding formulas.
    The function skips empty lines and lines without corresponding images.

    Args:
        image_path (str): The path to the directory containing the image files.
        math_file (str): The path to the text file containing the formulas.

    Returns:
        list, list: A list of image paths and a list of corresponding formula
    """
    image_names = [f for f in sorted(os.listdir(image_path))]
    image_paths = [os.path.join(image_path, f) for f in image_names]

    math_gts = []
    with open(math_file, 'r') as f:
        # load maths which
        for i, line in enumerate(f, start=1):
            image_name = f'{i - 1:07d}.png'
            if line.strip() and image_name in image_names:
                math_gts.append(line.strip())

    if len(image_paths) != len(math_gts):
        raise ValueError("The number of images does not match the number of formulas.")

    return image_paths, math_gts


def normalize_text(text):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, 1e8))
    bleu_results = bleu.compute(predictions=predictions, references=references)

    lev_dist = []
    for p, r in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return {
        'bleu': bleu_results["bleu"],
        'edit': sum(lev_dist) / len(lev_dist)
    }


def setup_seeds(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", default='data_process/configs/unimernet_base.yaml', help="path to configuration file.")
    parser.add_argument("--result_path", type=str, help="Path to json file to save result to.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main():
    setup_seeds()
    # Load Model and Processor
    start = time.time()
    cfg = Config(parse_args())
    task = tasks.setup_task(cfg)
    print(cfg)
    model = task.build_model(cfg)
    device = "cuda:0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    model.to(device)
    model.eval()

    print(f'arch_name:{cfg.config.model.arch}')
    print(f'model_type:{cfg.config.model.model_type}')
    print(f'checkpoint:{cfg.config.model.finetuned}')
    print(f'=' * 100)

    end1 = time.time()

    # Generate prediction with MFR model
    print(f'Device:{device}')
    print(f'Load model: {end1 - start:.3f}s')

    # Load Data (image and corresponding annotations)
    val_names = [
        "Simple Print Expression(SPE)",
        "Complex Print Expression(CPE)",
        "Screen Capture Expression(SCE)",
        "Handwritten Expression(HWE)"
    ]
    image_paths = [
        "./data/UniMER-Test/spe",
        "./data/UniMER-Test/cpe",
        "./data/UniMER-Test/sce",
        "./data/UniMER-Test/hwe"
    ]
    math_files = [
        "./data/UniMER-Test/spe.txt",
        "./data/UniMER-Test/cpe.txt",
        "./data/UniMER-Test/sce.txt",
        "./data/UniMER-Test/hwe.txt"
    ]

    # image_list, math_gts = load_data(image_path, math_file)

    transform = transforms.Compose([
        vis_processor,
    ])
    from pylatexenc.latex2text import LatexNodes2Text
    latex_nodes2text = LatexNodes2Text()
    # dataset = MathDataset(image_list, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=128, num_workers=32)
    data_dir = 'MLLM_data_3_v4/common_ratio_json_file_science_1010'
    save_path = os.path.join(os.path.dirname(data_dir), 'image_path_to_formular_text_1010.json')
    print(save_path)

    data = []
    for file_name in os.listdir(data_dir):
        data_path = os.path.join(data_dir, file_name)
        if not data_path.endswith('.json'):
            print(f'filter file={data_path}')
            continue
        with open(data_path, 'r') as f:
            data.extend(json.load(f))
    print(f'raw data num={len(data)}')
    image_list = []
    formular_image_data_num = 0
    for line in tqdm(data, total=len(data), desc='filter'):
        if 'image' in line:
            if isinstance(line['image'], str):
                image_path = line['image']
                image_path = os.path.join('raw_data', image_path)
                image = Image.open(image_path)
                if image.height < 50:
                    image_list.append(image_path)
                    formular_image_data_num += 1
            elif isinstance(line['image'], list):
                formular_data_flag = False
                for image_path in line['image']:
                    image_path = os.path.join('raw_data', image_path)
                    image = Image.open(image_path)
                    if image.height < 50:
                        image_list.append(image_path)

                        formular_data_flag = True

                if formular_data_flag:
                    formular_image_data_num += 1
    print(f'带公式图的数据量={formular_image_data_num}, 不带公示图的数据量={len(data) - formular_image_data_num}')
    print(f'formular image num={len(image_list)}')
    res_data = []


    transform = transforms.Compose([
        vis_processor,
    ])

    dataset = MathDataset(image_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=32)

    math_preds = []
    for images in tqdm(dataloader, total=len(dataloader), desc='predict'):
        images = images.to(device)
        with torch.no_grad():
            output = model.generate({"image": images})
        math_preds.extend(output["pred_str"])
    for model_pred, image_path in tqdm(zip(math_preds, image_list), total=len(image_list), desc='rm_latex'):
        try:
            text = latex_nodes2text.latex_to_text(model_pred)
            text = text.replace(' ', '')
        except:
            print(f'latex error:{model_pred}')
            text = ''
        temp_dict = {'image_path': image_path, 'model_pred': model_pred, 'clean_text': text}
        res_data.append(temp_dict)
    print(f'save to : {save_path}')
    with open(save_path, 'w') as f:
        json.dump(res_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
