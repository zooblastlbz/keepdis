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

# from DCGAN tutorial: according to GAN paper, model weights should be randomly
# initalized from mean 0  sd = 0.2; but this is for image classification, maybe
# we want something different for our purpose?

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_gan(args):
    device = 'cuda' # set device appropriately
    EPOCHS = 10
    G_losses = []
    D_losses = []

    ## boot up model and get everything running properly
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # get data
    d = CustomDataset(args.data)
    
    ds = {}
    ds["im_tok"] = d.im_toks
    ds["lang_tok"] = d.lang_toks

    tkn_lst = random.shuffle([ds["im_tok"],ds["lang_tok"]])
    
    # instantiate discriminator and send to device

    IMAGE_SIZE = 1024 * 5 # TODO: ensure this is correct
    NUM_CLASSES = 2
    discrim = Discriminator(IMAGE_SIZE, NUM_CLASSES)
    discrim.to(device)
    discrim.apply(weights_init)


    # instantiate generator and send to device


    
    for epoch in range(EPOCHS):
        for tkn in tkn_lst:
            # send token to discriminator; get prediciton

            # calculate loss of discriminator; somehow calculate loss of the generator 
            pass
        pass

    return NotImplemented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data, type=str, default=")
    args = parser.parse_args()

    train_gan(args) 