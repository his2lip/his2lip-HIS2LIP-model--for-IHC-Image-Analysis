import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torchvision
from transformers import CLIPProcessor, CLIPModel

#to fine tune Plip 

class PLIPTextEncoder(nn.Module):
    def __init__(self, model_name="vinid/plip", proj_dim=512, proj_bias=False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.projection_head = nn.Linear(512, proj_dim, bias=proj_bias)  # PLIP text embeddings are 512 by default

    def forward(self, input_ids, attention_mask):
        # Ensure input tensors are on GPU
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        # Extract text embeddings
        text_embeddings = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.projection_head(text_embeddings)
        return text_embeddings


class PLIPVisionEncoder(nn.Module):
    def __init__(self, model_name="vinid/plip", proj_dim=512, proj_bias=False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)  # Load pre-trained CLIP model
        self.projection_head = nn.Linear(512, proj_dim, bias=proj_bias)  # Project embeddings to shared space

    def forward(self, pixel_values):
        # Ensure pixel_values is on GPU
        if not pixel_values.is_cuda:
            pixel_values = pixel_values.cuda()

        # Extract vision embeddings
        img_embeddings = self.model.get_image_features(pixel_values=pixel_values)
        img_embeddings = self.projection_head(img_embeddings)
        return img_embeddings

# HIS22PLIP 

class HIS2PLIPModel(nn.Module):
    def __init__(self, model_name="vinid/plip", proj_dim=512):
        super().__init__()
        self.text_encoder = PLIPTextEncoder(model_name=model_name, proj_dim=proj_dim)
        self.vision_encoder = PLIPVisionEncoder(model_name=model_name, proj_dim=proj_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))  # Learnable temperature

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def encode_image(self, pixel_values):
        return self.vision_encoder(pixel_values)

    def forward(self, **inputs):
        # Extract pixel_values, input_ids, and attention_mask
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        # Encode text and image embeddings
        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Compute logits
        logits_per_image, logits_per_text = self.compute_logits(img_embeds, text_embeds)

        return {
            "img_embeds": img_embeds,
            "text_embeds": text_embeds,
            "logits": logits_per_image,
            "logits_per_text": logits_per_text,
        }

    def compute_logits(self, img_embeds, text_embeds):
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(img_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

#to fine tune clip 
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", proj_dim=512, proj_bias=False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.projection_head = nn.Linear(512, proj_dim, bias=proj_bias)  # CLIP text embeddings are 512 by default

    def forward(self, input_ids, attention_mask):
        # Ensure input tensors are on GPU
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        # Extract text embeddings
        text_embeddings = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.projection_head(text_embeddings)
        return text_embeddings


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", proj_dim=512, proj_bias=False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)  # Load pre-trained CLIP model
        self.projection_head = nn.Linear(512, proj_dim, bias=proj_bias)  # Project embeddings to shared space

    def forward(self, pixel_values):
        # Ensure pixel_values is on GPU
        if not pixel_values.is_cuda:
            pixel_values = pixel_values.cuda()

        # Extract vision embeddings
        img_embeddings = self.model.get_image_features(pixel_values=pixel_values)
        img_embeddings = self.projection_head(img_embeddings)
        return img_embeddings


class HIS2CLIPModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", proj_dim=512):
        super().__init__()
        self.text_encoder = CLIPTextEncoder(model_name=model_name, proj_dim=proj_dim)
        self.vision_encoder = CLIPVisionEncoder(model_name=model_name, proj_dim=proj_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))  # Learnable temperature

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def encode_image(self, pixel_values):
        return self.vision_encoder(pixel_values)

    def forward(self, **inputs):
        # Extract pixel_values, input_ids, and attention_mask
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        # Encode text and image embeddings
        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Compute logits
        logits_per_image, logits_per_text = self.compute_logits(img_embeds, text_embeds)

        return {
            "img_embeds": img_embeds,
            "text_embeds": text_embeds,
            "logits": logits_per_image,
            "logits_per_text": logits_per_text,
        }

    def compute_logits(self, img_embeds, text_embeds):
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(img_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


class PromptClassifier(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

