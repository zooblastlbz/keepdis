import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random


class Discriminator(nn.Module):
    def __init__(self, input_size, learning_rate=0.0002, adam_beta1=0.9):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def linear(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
    
    def equal_sample(self, image, language): 
        min_size = min(image.size(0), language.size(0))
        image = image[:min_size]
        language = language[:min_size]

        return image, language

    def run_forward(self, data, d_mode):

        image_list = torch.unbind(data['image'][0], dim=0)

        zipped_lists = list(zip(image_list, data["lang"]))
        total_loss = 0
        img_is_correct = 0
        lang_is_correct = 0


        for image, language in zipped_lists:
            # can add some logic to balance the dataset? usually lang tokens are less than image
            image, language = self.equal_sample(image, language)
            if d_mode:
                loss, img, lang = self.forward(
                    image.view(-1, 5120), language.view(-1, 5120), d_mode
                )
                total_loss += loss/len(zipped_lists)
                img_is_correct += torch.sum(img)
                lang_is_correct += torch.sum(lang)

            else:
                total_loss += self.forward(image.view(-1, 5120), language.view(-1, 5120), d_mode) / len(zipped_lists)

        # for debug
        print("dmode: ", d_mode, "image is correct: ", img_is_correct)
        print("dmode: ", d_mode, "lang is correct: ", lang_is_correct)

        if d_mode:
            return {
                "loss": total_loss,
                "img_is_correct": img_is_correct,
                "lang_is_correct": lang_is_correct,
            }

        return total_loss

    def forward(self, img_tok, lang_tok, d_mode):

        loss_function = nn.BCELoss()  # from DCGAN

        img_pred = self.linear(img_tok)
        lang_pred = self.linear(lang_tok)

        if d_mode == True:

            img_label = torch.full(
                (img_tok.size(0), 1), 1, dtype=torch.bfloat16, device=self.device
            )  # 1 for images
            lang_label = torch.full(
                (lang_tok.size(0), 1), 0, dtype=torch.bfloat16, device=self.device
            )  #  0 for lang

            img_loss = loss_function(img_pred, img_label)
            lang_loss = loss_function(lang_pred, lang_label)

            loss = img_loss + lang_loss

            img_pred_binary = torch.ge(img_pred, 0.5).float().to(torch.bfloat16)
            lang_pred_binary = torch.ge(lang_pred, 0.5).float().to(torch.bfloat16)
            # >= because we want the tensor to be all 0s if each value is less than 0.5

            img_is_correct = torch.eq(img_pred_binary, img_label)
            lang_is_correct = torch.eq(lang_pred_binary, lang_label)

            return loss, img_is_correct, lang_is_correct

        else:
            lang_label = torch.full(
                (img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=self.device
            )  #  0 for lang
            img_with_lang_label_loss = loss_function(
                img_pred, lang_label
            )  # trying to follow DCGAN

            return img_with_lang_label_loss  # returning image loss to maximize disc loss when training generator
