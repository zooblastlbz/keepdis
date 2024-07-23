import argparse
import torch
import torch.nn as nn
from PIL import Image
import json
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import math
import os
import random
from datetime import datetime

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
#from llava.model.llava_arch import prepare_inputs_labels_for_multimodal

from discriminator import preprocess_and_call_train

def split_list(lst, n): # taken from model_vqa.py
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k): # taken from model_vqa.py
    chunks = split_list(lst, n)
    return chunks[k]

def get_tkns(input_ids, image_tensor, model, img_size):
    '''takes in one prompt and one image and  returns a dictionary that has the language tokens
    from the prompt as one entry and the image tokens from the prompt as another'''
    position_ids = None # set to None in generate()
    attention_mask = None # set to None in generate() must figure out if this and the above is acceptable

    # prep_inputs... returns None as the first value, but idk why
    none_q, position_ids, attention_mask, past_key_values, input_embeds, labels, chunk_sizes = model.prepare_inputs_labels_for_multimodal(
        input_ids = input_ids,
        position_ids = position_ids,
        attention_mask = attention_mask,
        past_key_values = None,
        labels = None,
        images = image_tensor.unsqueeze(0).half().cuda(),
        image_sizes = img_size
    )

    split_embeddings = torch.split(input_embeds[0], chunk_sizes, dim=0)
    lang_tkns = split_embeddings[2] # only the second to avoid adding the same tokens over and over
    img_tkns = split_embeddings[1]

    tkn_dict = {
        lang_tkns: lang_tkns,
        img_tkns: img_tkns
    }

    return tkn_dict

def prep_batches(line, model, tokenizer, image_processor, rags, **kwargs):
    q_id = line["id"] # can be used to identify each batch, probably good to use to keep track of progress during training
    image_file = line["image"]
    qs = line["text"]

    if qs.startswith(f"{DEFAULT_IMAGE_TOKEN}\n") == False:
        idx = qs.find(DEFAULT_IMAGE_TOKEN) + len(DEFAULT_IMAGE_TOKEN)
        qs = qs[idx:].strip()
        qs = qs[idx:]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        assert qs.startswith(f"{DEFAULT_IMAGE_TOKEN}\n") == True, f'no image tag found in text \n text = {qs} \n id = {q_id}'

    # something to note: this appends a default prompt to each prompt, might impact discrim since it will keep getting trained on
    # the same tokens. i'll adjust to remove this soon

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_sizes = [image.size]

    tkn_dict = get_tkns(input_ids, image_tensor, model, image_sizes) #returns tkn_dict with image and language tokens

    projection_model = preprocess_and_call_train(tkn_dict)

def train(args):
    args_dict = vars(args)

    device = 'cuda' # set device appropriately
    EPOCHS = 10
    G_losses = []
    D_losses = []
    iters = 0

    ## boot up model and get everything running properly
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # get data - following along with model_vqa.py
    questions = [json.loads(q) for q in open(os.path.expanduser(args.conversation_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # right now each batch is created one by one for each conversation in the file, maybe we want to precompute all the
    # batches ahead of time?  maybe we consolidate this for-loop into a function? for now it should work but
    # just some things to think about

    for line in questions:
        tkn_dict = prep_batches(line, model, tokenizer, image_processor, args, **args_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default= "/home/smirrashidi/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default= "/home/smirrashidi/coco_data/images")
    parser.add_argument("--conversation_file", type=str, default= "/home/smirrashidi/coco_data/discrim_data.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    train(args)
