import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()  # we can add more layers later

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)

    def linear(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
 
    def forward(self, data, d_mode):
        device = 'cuda'   # CHANGE TO JUST CUDA WHEN NOT USING VLMEVAL
        loss_function = nn.BCELoss() # from DCGAN

        img_tok = data["image"]
        lang_tok = data["lang"]

        img_tok = img_tok.view(-1, 5120) # image tokens have dim=3

        img_pred = self.linear(img_tok)
        lang_pred = self.linear(lang_tok)

        if d_mode == True: 

            img_label = torch.full((img_tok.size(0), 1), 1, dtype=torch.bfloat16, device=device)  # 1 for images
            lang_label = torch.full((lang_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device)  #  0 for lang

            img_loss = loss_function(img_pred, img_label)
            lang_loss = loss_function(lang_pred, lang_label) 

            loss = img_loss + lang_loss

            img_pred_binary = torch.ge(img_pred, 0.5).float().to(torch.bfloat16)
            lang_pred_binary = torch.ge(lang_pred, 0.5).float().to(torch.bfloat16) # >= because we want the tensor to be all 0s if each value is less than 0.5
    
            img_is_correct = torch.eq(img_pred_binary, img_label)    
            lang_is_correct = torch.eq(lang_pred_binary, lang_label)
                        
            return_dict = {
                "loss": loss, 
                "img_is_correct" : img_is_correct, 
                "lang_is_correct": lang_is_correct, 
            }

            return return_dict
        
        else:
            lang_label = torch.full((img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device)  #  0 for lang
            img_with_lang_label_loss = loss_function(img_pred, lang_label) # trying to follow DCGAN

            return img_with_lang_label_loss # returning image loss to maximize disc loss when training generator
        