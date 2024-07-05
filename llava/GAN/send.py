import argparse
import torch
import os
import random
from datetime import datetime

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from discriminator.py import Discriminator # import Laya's discriminator
from make_data.py import CustomDataset # impor the dataset class

def get_data(image_folder, language_file):
    '''takes in images and the language file and outputs a shuffled list
    of both the images after going through _projector and the tokenized language tokens'''

    return None

def send_to_discriminator():
    ## boot up model and get everything running properly
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data, type=str, default=")
    args = parser.parse_args()