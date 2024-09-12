from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.run_llava import load_images, image_parser
import re
import torch
import torch.nn as nn
import tflite_runtime.interpreter as tflite

from transformers import AutoTokenizer, CLIPImageProcessor, LlamaForCausalLM
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.multimodal_encoder.mobilenetv2_encoder import MobileNetV2VisionTower
import os
import numpy as np

torch.set_printoptions(threshold=10000)

# Lora model path or Full model path
model_path = "./checkpoints/llava-v1.5-13b-lora-v2/checkpoint-200"
model_name = get_model_name_from_path(model_path)
# Base model path if model_path is lora-based, otherwise None
base_model_path = "lmsys/vicuna-13b-v1.5"

# Query
question = "What happened?"
# Image path
image_path = "./blip_laion/00000/000000012.jpg"

# Generation parameters
temperature = 0
top_p=None
num_beams=1
max_new_tokens=512

lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)

if image_path is None:
    def encode_image(vision_tower, height, width):
        noise_image = np.random.normal(loc=0.5, scale=0.2, size=(height, width, 3))
        noise_image = np.clip(noise_image, 0, 1)
        noise_image = (noise_image * 255).astype(np.uint8)
            
        encoder = MobileNetV2VisionTower(vision_tower)
        
        image = Image.fromarray(noise_image)
        image = encoder.image_processor([image], return_tensors='pt')['pixel_values']
        
        return encoder.process_single_image(image.to(dtype=torch.float16))
    
    print('Encode random gaussian noise...')
    image_embeddings = encode_image(
        getattr(lora_cfg_pretrained, "mm_vision_tower", None),
        1024,
        1024,
    )
else:
    def encode_image(vision_tower, image_path):
        encoder = MobileNetV2VisionTower(vision_tower)
        
        image = Image.open(image_path).convert("RGB")
        image = encoder.image_processor([image], return_tensors='pt')['pixel_values']
        
        return encoder.process_single_image(image.to(dtype=torch.float16))

    print('Encode image...')
    image_embeddings = encode_image(
        getattr(lora_cfg_pretrained, "mm_vision_tower", None),
        image_path,
    )

def build_vision_projector(projector_type, mm_hidden_size, hidden_size):
    if projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

print('Build projector...')
mm_projector = build_vision_projector(
    getattr(lora_cfg_pretrained, "mm_projector_type", None),
    getattr(lora_cfg_pretrained, "mm_hidden_size", None),
    getattr(lora_cfg_pretrained, "hidden_size", None),
).to(dtype=torch.float16).cuda()

print('Load lora trainables...')
non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location=image_embeddings.device)
non_lora_trainables = {(k[17:] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}

print('Split and load projector trainables...')
mm_projector_trainables = {(k[19:] if k.startswith('model.mm_projector.') else k): v for k, v in non_lora_trainables.items()}
mm_projector.load_state_dict(mm_projector_trainables, strict=False)

print('Loading Base LLM...')
model = LlamaForCausalLM.from_pretrained(base_model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained).to(dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

print('Loading LoRA weights...')
from peft import PeftModel
model = PeftModel.from_pretrained(model, model_path)

print('Merging LoRA weights...')
model = model.merge_and_unload()

print('Model is loaded...')

# Project image embeddings to LLM compatiable embeddings
image_features = mm_projector(image_embeddings)

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

print(f'Conversation template {conv_mode} is used...')
conv = conv_templates[conv_mode].copy()

# Combine question with image
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize prompt
input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)

# Convert image tag to image embedding, embed non-image tokens and combine into one embeddings
### Copied from prepare_inputs_labels_for_multimodal in llava/model/llava_arch.py ###
labels = torch.full_like(input_ids, IGNORE_INDEX)
new_input_embeds = []
new_labels = []
cur_image_idx = 0
for batch_idx, cur_input_ids in enumerate(input_ids):
    num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    if num_images == 0:
        cur_image_features = image_features[cur_image_idx]
        cur_input_embeds_1 = model.model.embed_tokens(cur_input_ids)
        cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
        new_input_embeds.append(cur_input_embeds)
        new_labels.append(labels[batch_idx])
        cur_image_idx += 1
        continue

    image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    cur_input_ids_noim = []
    cur_labels = labels[batch_idx]
    cur_labels_noim = []
    for i in range(len(image_token_indices) - 1):
        cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
        cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    split_sizes = [x.shape[0] for x in cur_labels_noim]
    cur_input_embeds = model.model.embed_tokens(torch.cat(cur_input_ids_noim))
    cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    cur_new_input_embeds = []
    cur_new_labels = []

    for i in range(num_images + 1):
        cur_new_input_embeds.append(cur_input_embeds_no_im[i])
        cur_new_labels.append(cur_labels_noim[i])
        if i < num_images:
            cur_image_features = image_features[cur_image_idx]
            cur_image_idx += 1
            cur_new_input_embeds.append(cur_image_features)
            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    cur_new_input_embeds = [x.to(image_embeddings.device) for x in cur_new_input_embeds]

    cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    cur_new_labels = torch.cat(cur_new_labels)

    new_input_embeds.append(cur_new_input_embeds)
    new_labels.append(cur_new_labels)

# Truncate sequences to max length as image embeddings can make the sequence longer
tokenizer_model_max_length = getattr(lora_cfg_pretrained, 'tokenizer_model_max_length', None)
if tokenizer_model_max_length is not None:
    new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

# Combine them
max_len = max(x.shape[0] for x in new_input_embeds)
batch_size = len(new_input_embeds)

new_input_embeds_padded = []
new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)

for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    cur_len = cur_new_embed.shape[0]
    if getattr(lora_cfg_pretrained, 'tokenizer_padding_side', 'right') == "left":
        new_input_embeds_padded.append(torch.cat((
            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
            cur_new_embed
        ), dim=0))
        if cur_len > 0:
            new_labels_padded[i, -cur_len:] = cur_new_labels
    else:
        new_input_embeds_padded.append(torch.cat((
            cur_new_embed,
            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
        ), dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels

new_input_embeds = torch.stack(new_input_embeds_padded, dim=0).detach()
###

# Generate answer
output_data = model.generate(
    position_ids=None,
    attention_mask=None,
    inputs_embeds=new_input_embeds,
    do_sample=True if temperature > 0 else False,
    temperature=temperature,
    top_p=top_p,
    num_beams=num_beams,
    max_new_tokens=max_new_tokens,
    use_cache=True,
)

# Decode answer
outputs = tokenizer.batch_decode(output_data, skip_special_tokens=True)[0].strip()

# Print answer
print("PROMPT:")
print(prompt)
print("ANSWER:")
print(outputs)