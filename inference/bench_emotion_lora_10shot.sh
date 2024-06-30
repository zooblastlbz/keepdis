#!/bin/bash

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ngpus=2:ncpus=16

### Specify amount of time required
#PBS -l walltime=10:00:00

### Specify project code
#PBS -P 12002486

### Specify name for job
#PBS -N bench_emotion_lora_10shot

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### For automatic mailing, use the following options:
#PBS -M chih0001@e.ntu.edu.sg
#PBS -m abe

### Start of commands to be run
# 获取当前日期和时间，格式为 YYYYMMDD-HHMMSS
NOW=$(date +%Y%m%d-%H%M%S)

# 将标准输出和错误重定向到包含日期时间的文件名
exec > /home/users/ntu/chih0001/scratch/VLM/SGLang/benchmark/llava_bench/inference_logs/${PBS_JOBNAME}_${NOW}.o${PBS_JOBID} 2>&1

source /home/users/ntu/chih0001/anaconda3/etc/profile.d/conda.sh
conda activate llava-test

export CUDA_VISIBLE_DEVICES=0,1

cd /home/users/ntu/chih0001/scratch/VLM/SGLang/benchmark/llava_bench

python -m llava.eval.model_vqa \
    --model-path /home/users/ntu/chih0001/scratch/model/lora/globalBS/llava-v1.5-7b-DSLLM-lora_10shot \
    --model-base /home/users/ntu/chih0001/scratch/model/llava-v1.5-7b \
    --question-file /home/users/ntu/chih0001/scratch/VLM/SGLang/benchmark/llava_bench/questions/emotion/questions_FED.jsonl \
    --image-folder /home/users/ntu/chih0001/scratch/data/emotion/KMU_FED/test \
    --answers-file /home/users/ntu/chih0001/scratch/VLM/SGLang/benchmark/llava_bench/answers/FED/lora_10shot_answers_FED_1.5-7b_new.jsonl \
    --conv-mode vicuna_v1 \
    --temperature 0 