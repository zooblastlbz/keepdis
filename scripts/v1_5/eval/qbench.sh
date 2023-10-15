#!/bin/bash


python -m llava.eval.model_vqa_qbench \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-folder ./playground/data/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_$1.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_$1_answers.jsonl \
    --conv-mode llava_v1 \
    --lang en
