python model_vqa.py \
    --model-path /home/wxl/xyb/LLaVA/base_model/llava-v1.6-vicuna-7b \
    --question-file /home/wxl/xyb/LLaVA/playground/data/5shot/eval_bentch_cot.json \
    --image-folder /home/wxl/xyb/LLaVA/playground/data/5shot/img_test \
    --answers-file /home/wxl/xyb/LLaVA/playground/data/test_llava-v1.6-vicuna-7b-chunk.jsonl \
    --conv-mode llava_v1