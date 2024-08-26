import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid  # 用于生成短的UUID

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle  # 用于对话模版和分隔符样式
from llava.model.builder import load_pretrained_model  # 用于加载预训练模型
from llava.utils import disable_torch_init  # 用于禁用torch的初始化的实用工具
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path  # 用多模态处理的工具

from PIL import Image  
import math  


def split_list(lst, n):
    """
    Split a list into n (roughly) equal-sized chunks
    将一个列表 lst 分割成 n 个大致相等大小的块

    Args:
    - lst: 要分割的列表
    - n: 分割的块数

    Returns:
    - list: 分割后的列表
    """
    chunk_size = math.ceil(len(lst) / n)  # 确定每个块的大小，使用 math.ceil 函数来计算每个块的大小，将列表的长度除以 n 并向上取整
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]  # 使用列表推导式，根据计算出的 chunk_size 将 lst 切分为若干块，每块包含 chunk_size 个元素


def get_chunk(lst, n, k):
    """
    Get the k-th chunk of a list split into n chunks
    获取将列表分成 n 个块后的第 k 个块
    
    Args:
    - lst: 要分割的列表
    - n: 分割的块数
    - k: 要获取的块的索引

    Returns:
    - list: 分割后的列表的第 k 个块
    """
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    """
    基于给定的的参数评估预训练模型

    Args:
    - args: 参数

    Returns:
    - None
    """
    # Model
    disable_torch_init()  # 禁用 torch 的初始化
    model_path = os.path.expanduser(args.model_path)  # 扩展用户路径，获取模型路径
    model_name = get_model_name_from_path(model_path)  # 从模型路径中获取模型名称
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)  # 加载预训练模型和相关组件

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]  # 读取并解析问题数据文件，生成questions列表
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)  # 将问题数据分成 args.num_chunks 个块，获取第 args.chunk_idx 个块
    answers_file = os.path.expanduser(args.answers_file)  # 扩展用户路径，获取答案文件路径
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)  # 创建答案文件的目录
    ans_file = open(answers_file, "w")  # 打开答案文件，准备写入
    for line in tqdm(questions):
        # 1. 提取问题ID、图像文件和文本内容
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text1"]
        qs_text2 = line["text2"]
        cur_prompt = qs
        cur_prompt_text2 = qs_text2
        

        # 2. 处理问题文本，根据模型配置决定是否在问题文本前后添加特定标记
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 3. 构建对话模版
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 4. 构建Cot代码
        if model.config.mm_use_im_start_end:
            qs_text2 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_text2
        else:
            qs_text2 = qs_text2
        
        conv.append_message(conv.roles[1], outputs)
        conv.append_message(conv.roles[0], qs_text2)
        conv.append_message(conv.roles[1], None)
        prompt_text2 = conv.get_prompt()

        input_ids2 = tokenizer_image_token(prompt_text2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids2 = model.generate(
                input_ids2,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs_text2 = tokenizer.batch_decode(output_ids2, skip_special_tokens=True)[0].strip()

        # 5. 生成答案ID并写入答案文件
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt_text1": cur_prompt,
                                   "text1": outputs,
                                   "prompt_text2": cur_prompt_text2,
                                   "text2": outputs_text2,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

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
