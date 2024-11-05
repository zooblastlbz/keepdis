import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import *
from llava.train.llava_trainer import LLaVATrainer

from PIL import Image
import math
from llava.VLLMSafety.discriminator import Discriminator
from datetime import date

class DiscArguments:
    test_data_path: str = "/home/smirrashidi/coco_data/coco_test_conversations.json"
    test_image_folder: str = "/home/smirrashidi/coco_data/coco_test"
    model_path: str = "/home/smirrashidi/LLaVAFork/checkpoints/llava-v1.5-13b-lora-disc"
    model_base: str = "lmsys/vicuna-13b-v1.3"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, disc_args, testing) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    if testing == True:
        data_args.image_folder = "/home/smirrashidi/coco_data/coco_test"
        data_args.data_path = "/home/smirrashidi/coco_data/coco_test_conversations.json"

        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    else:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
    return dict(train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_args = ModelArguments(model_name_or_path = args.model_path)
    data_args = DataArguments(data_path = args.question_file, 
                image_folder = args.image_folder)
    training_args = TrainingArguments(output_dir="/home/smirrashidi/dump")
    disc_args = DiscArguments

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model = model.to(torch.bfloat16)

    data_args.image_processor = image_processor

    test_data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args, 
                                              disc_args=disc_args, 
                                              testing = True)

    data_collator = test_data_module['data_collator']

    test_dataloader = DataLoader(
        test_data_module['train_dataset'],  
        batch_size=4,  
        collate_fn=data_collator, 
        shuffle=False)

    eval_disc_data = {
                "num_img_corr": 0,
                "num_lang_corr": 0,
                "img_total": 0,
                "lang_total": 0,
            }

    for i, batch in enumerate(test_dataloader):
        print(f"Iteration #{i}")
        input_ids = batch['input_ids']
        image = batch['images']
        with torch.inference_mode():
            discrim_dict = model.forward_eval_discrim(
                input_ids = input_ids, 
                images = image
            )

        eval_disc_data["num_img_corr"] += discrim_dict["img_is_correct"].sum().item()
        eval_disc_data["num_lang_corr"] += discrim_dict["lang_is_correct"].sum().item()
        eval_disc_data["img_total"] += discrim_dict["img_is_correct"].size(0)
        eval_disc_data["lang_total"] += discrim_dict["lang_is_correct"].size(0)
        
    eval_disc_data["date"] = date.today().strftime('%Y-%m-%d')
    print(eval_disc_data)

    with open("/home/smirrashidi/eval_discrim_results.json", "a") as json_file:
        json.dump(eval_disc_data, json_file)
        json_file.write("\n") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/smirrashidi/LLaVAFork/checkpoints/llava-v1.5-13b-lora-disc")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--image-folder", type=str, default="/home/smirrashidi/coco_data/coco_test")
    parser.add_argument("--question-file", type=str, default="/home/smirrashidi/coco_data/coco_test_conversations.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)

