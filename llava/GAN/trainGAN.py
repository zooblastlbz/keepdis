import argparse
import torch
import torch.nn as nn
import json
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import os
import random
from datetime import datetime

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from discriminator.py import Discriminator # import Laya's discriminatorcod
from make_data.py import CustomDataset # impor the dataset class

# from DCGAN tutorial: according to GAN paper, model weights should be randomly
# initalized from mean 0  sd = 0.2; but this is for image classification, maybe
# we want something different for our purpose?

def split_list(lst, n): # taken from model_vqa.py
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k): # taken from model_vqa.py
    chunks = split_list(lst, n)
    return chunks[k]

def prep_convo_for_discrim(prompt, image):
    '''takes in one prompt and one image and  returns a dictionary that has the language tokens 
    from the prompt as one entry and the image tokens from the prompt as another'''

def train_gen(args):
    device = 'cuda' # set device appropriately
    EPOCHS = 10
    G_losses = []
    D_losses = []
    iters = 0 

    ## boot up model and get everything running properly - THIS IS THE GENERATOR
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # get data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.data), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    for line in questions:
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        curr_prompt = qs 
        




    real_label = 1
    fake_label = 0
    
    
    
    
    lr = 0.0002
    beta1 = 0.5

    # instantiate discriminator and send to device

    IMAGE_SIZE = 1024 * 5 # TODO: ensure this is correct
    NUM_CLASSES = 2
    discrim = Discriminator(IMAGE_SIZE, NUM_CLASSES) # TODO: apply weights
    discrim.to(device)

    # TODO: instantiate generator and send to device - i dont think this code is correct, maybe its better to load the weights directly into a new class 
    mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    gen = model.load_state_dict(mm_projector_weights, strict=False)

    # loss functions and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discrim.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999)) 
    
    for epoch in range(EPOCHS):
        # TODO: do we train the discrim at all? or simply send generated tokens to discrim and calcualte loss
        for i, (data1, data0) in enumerate(zip(dataloader1, dataloader0)): # this loop is following along DCGAN tutorial in pytroch documentation

            # train discriminator with all real batch
            discrim.zero_grad()

            real_batch = data1[0].to(device)
            rb_size = real_batch.size(0)
            label = torch.full((rb_size,), real_label, dtype=torch.float, device=device)

            # forward pass through discrim
            output = discrim(real_batch).view(-1)

            # calculate loss on real token batch
            errD_real = criterion(output, label)

            #calculate gradients backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # train discriminator with all fake batch
            fake_batch = data0[0].to(device)
            fb_size = fake_batch.size(0)
            label = torch.full((fb_size,), fake_label, dtype=torch.float, device=device)
            
            # project the tokens 
            fake_tkns = gen(fake_batch)

            # send fake batch to discrim for classification
            output = discrim(fake_tkns.detach()).view(-1)

            # calculate discrim loss on fake batch 
            errD_fake = criterion(output, label)
            D_G_z1 = output.mean().item()

            # compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # update D
            optimizerD.step()



            pass
        pass

    return NotImplemented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data, type=str, default=")
    args = parser.parse_args()

    train_gen(args) 