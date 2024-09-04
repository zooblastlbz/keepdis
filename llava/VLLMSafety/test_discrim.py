import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset, make_supervised_data_module

from PIL import Image
import math
from llava.VLLMSafety.discriminator import Discriminator


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    total = 0 
    num_img_correct= 0
    num_lang_correct = 0

    saved_model = Discriminator(5120)
    saved_model.load_state_dict(torch.load("/home/smirrashidi/LLaVAFork/trained_disc_copy"), strict=False)
    
    model.discriminator = saved_model
    

        with torch.inference_mode():
            discrim_dict = model.forward_eval_discrim(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                saved_model = saved_model
                )
            
            losses.append(discrim_dict["loss"])
            img_is_correct = discrim_dict["img_is_correct"]
            lang_is_correct = discrim_dict["lang_is_correct"] 

        if img_is_correct == True:
            num_img_correct += 1
        
        if lang_is_correct == True:
            num_lang_correct += 1

        total += 1
    
    img_accuracy = num_img_correct / total
    lang_accuracy = num_lang_correct / total

    print(f'Image Accuracy: {img_accuracy} \n Lang Accuracy: {lang_accuracy} \n')

    return losses

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

