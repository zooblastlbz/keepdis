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
        device = 'cuda'  
        loss_function = nn.BCELoss()  # follow DCgan

        image_batch = data['image'][0].view(-1, 5120).to(device)
        img_tok = image_batch.view(-1, 5120)  # flatten the lists

        img_pred = self.linear(img_tok)
        img_label = torch.full((img_tok.size(0), 1), 1, dtype=torch.bfloat16, device=device)  # use label 1 for imgs
        img_loss = loss_function(img_pred, img_label)

        total_lang_loss = 0
        lang_correct_count = 0
        total_lang_preds = 0
        img_correct_count = torch.eq(torch.ge(img_pred, 0.5).float().to(torch.bfloat16), img_label).sum().item()
        img_accuracy = img_correct_count / img_tok.size(0) * 100

        for lang_tensor in data["lang"]:
            lang_tensor = lang_tensor.to(device)
            lang_pred = self.linear(lang_tensor.view(-1, 5120))  # Process each lang tensor independently
            lang_label = torch.full((lang_pred.size(0), 1), 0, dtype=torch.bfloat16, device=device)  # Label 0 for language

            lang_loss = loss_function(lang_pred, lang_label)
            total_lang_loss += lang_loss

            #for accuracy calculations
            lang_correct = torch.eq(torch.ge(lang_pred, 0.5).float().to(torch.bfloat16), lang_label).sum().item()
            lang_correct_count += lang_correct
            total_lang_preds += lang_pred.size(0)

        if d_mode:
            lang_accuracy = lang_correct_count / total_lang_preds * 100
            print(f"Image Accuracy: {img_accuracy:.2f}%")
            print(f"Language Accuracy: {lang_accuracy:.2f}%")

            loss = img_loss + total_lang_loss

            return {
                "loss": loss, 
                "img_is_correct": img_correct_count, 
                "lang_is_correct": lang_correct_count, 
                "img_accuracy": img_accuracy, 
                "lang_accuracy": lang_accuracy,
            }
        
        else:
            img_with_lang_label_loss = loss_function(img_pred, torch.full((img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device))
            return img_with_lang_label_loss