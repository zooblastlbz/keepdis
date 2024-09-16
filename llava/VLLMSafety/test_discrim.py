import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.VLLMSafety.builder2 import load_pretrained_model_discrim
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset, DataArguments, TrainingArguments, make_supervised_data_module
from llava.train.llava_trainer import LLaVATrainer

from PIL import Image
import math
from llava.VLLMSafety.discriminator import Discriminator
from llava.train.train import *


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model_discrim(model_path, args.model_base, model_name)
        model_args = ModelArguments(model_name_or_path = args.model_path)
        data_args = DataArguments(data_path = args.question_file, 
                    image_folder = args.image_folder)
    
        
        training_args = TrainingArguments(output_dir="/home/smirrashidi/dump")

        
        total = 0 
        num_img_correct= 0
        num_lang_correct = 0
    
        # test_data = LazySupervisedDataset(tokenizer=tokenizer,
        #                     data_path=data_args.data_path,
        #                     data_args=data_args)

        # data_module = make_supervised_data_module(tokenizer=tokenizer,
        #                                         data_args=data_args)

        # trainer = LLaVATrainer(model=model,
        #                 tokenizer=tokenizer,
        #                 args=training_args,
        #                 **data_module)
        
        # trainer.evaluate(eval_dataset=test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)

