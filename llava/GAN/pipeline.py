import argparse
import torch
import os
import random
import json


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from discriminator.py import discriminator # import Laya's discriminator

def get_data(image_folder, language_file):
    '''takes in a folder of images and returns the tokens - images go trough both the clip encoder and the mm_projector; also takes in a language file and gets those toekens'''
    return NotImplemented

def train_gan(args):
    EPOCHS = 10
    G_losses = []
    D_losses = []

    ## boot up model and get everything running properly
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for epoch in range(EPOCHS):

        # how to decide whether to pass an image or language token to the discriminator
        tkn_type = 0 if (random.random() % 2 == 0) else 1

        # if image - use prepare inputs for mulitmotdal with only one image; or use encode_images 
        if tkn_type == 1:
            
            image_tkn = 
        # if language - use tokenizer_image_token

        # run discriminator; get prediciton

        # calculate loss of discriminator; somehow calculate loss of the generator 


    return NotImplemented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--language-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    train_gan(args) 