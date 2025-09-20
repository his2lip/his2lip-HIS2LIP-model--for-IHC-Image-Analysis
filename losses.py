from torch import nn
import torch.nn.functional as F
import torch
import pdb
import numpy as np

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        aug_input_ids=None,
        aug_attention_mask=None,
        **kwargs,
        ):
        '''args:
        labels: the image corresponds to which classes of diagnoses
        text_labels: the text corresponds to which classes of diagnoses
        '''
        if img_labels is None or text_labels is None:
            '''use hard clip loss as the original clip
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=True,
                    )
        else:
            '''use soft clip loss
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=False,
                    )

            # get logits
            logits = outputs['logits']

            # compute soft-labels, -1: negative, 0: uncertain, 1: positive
            # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
            label_sim = torch.matmul(img_labels, text_labels.T)
            label_sim = label_sim.to(logits.device)

            if aug_input_ids is not None:
                aug_text_embeds = self.model.encode_text(aug_input_ids, aug_attention_mask)
                img_embeds = outputs['img_embeds']
                logits_aug = self.model.compute_logits(img_embeds, aug_text_embeds)
                aug_loss_value = self._soft_clip_loss(logits_aug, label_sim)
                loss_value = self._soft_clip_loss(logits, label_sim)
                outputs['loss_value'] = (aug_loss_value + loss_value) / 2
            else:
                outputs['loss_value'] = self._soft_clip_loss(logits, label_sim)

        return_res = {
            'loss_value': outputs['loss_value'],
        }
        return return_res

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(logits_per_img, F.softmax(soft_label,1))
        caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        return (image_loss + caption_loss) / 2

        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        # return image_loss

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]

    def _soft_bce_loss(self, input, target):
        return nn.functional.binary_cross_entropy_with_logits(input, target)

class CLIPLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.logit_scale = nn.Parameter(torch.ones([]))  # Initialize logit scale

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        # Encode text and image features
        text_features = self.model.encode_text(input_ids, attention_mask)
        image_features = self.model.encode_image(pixel_values)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Compute losses
        caption_loss = nn.functional.cross_entropy(
            logits_per_image, torch.arange(len(logits_per_image)).to(logits_per_image.device)
        )
        image_loss = nn.functional.cross_entropy(
            logits_per_text, torch.arange(len(logits_per_text)).to(logits_per_text.device)
        )

        # Combine losses
        loss_value = (caption_loss + image_loss) / 2
        return {"loss_value": loss_value}

        
import torch
import torch.nn as nn

class ImageToImageContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values=None, img_labels=None, **kwargs):
        # Encode image embeddings
        img_embeds = self.model.encode_image(pixel_values)

        # Normalize embeddings
        img_embeds = torch.nn.functional.normalize(img_embeds, dim=1)

        # Compute similarity logits
        logits = torch.matmul(img_embeds, img_embeds.T)

        # Compute soft-label similarity
        label_sim = torch.matmul(img_labels, img_labels.T)
        label_sim = label_sim.to(logits.device)

        # Calculate loss
        loss_value = self._soft_xent_loss(logits, torch.nn.functional.softmax(label_sim, dim=1))
        return {"loss_value": loss_value}

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]


class TextToTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, text_labels=None, **kwargs):
        # Encode text embeddings
        text_embeds = self.model.encode_text(input_ids, attention_mask)

        # Normalize embeddings
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=1)

        # Compute similarity logits
        logits = torch.matmul(text_embeds, text_embeds.T)

        # Compute soft-label similarity
        label_sim = torch.matmul(text_labels, text_labels.T)
        label_sim = label_sim.to(logits.device)

        # Calculate loss
        loss_value = self._soft_xent_loss(logits, torch.nn.functional.softmax(label_sim, dim=1))
        return {"loss_value": loss_value}

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]


class CombinedContrastiveLoss(nn.Module):
    def __init__(self, model, weights=None):
        super().__init__()
        self.model = model
        self.weights = weights if weights else {"image_text": 1.0, "image_image": 0.5, "text_text": 0.5}

        # Initialize sub-losses
        self.image_text_loss = ImageTextContrastiveLoss(model)
        self.image_image_loss = ImageToImageContrastiveLoss(model)
        self.text_text_loss = TextToTextContrastiveLoss(model)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        **kwargs,
    ):
        # Compute individual losses
        image_text_output = self.image_text_loss(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            img_labels=img_labels,
            text_labels=text_labels,
        )

        image_image_output = self.image_image_loss(
            pixel_values=pixel_values,
            img_labels=img_labels,
        )

        text_text_output = self.text_text_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_labels=text_labels,
        )

        # Combine the losses using weights
        combined_loss = (
            self.weights["image_text"] * image_text_output["loss_value"] +
            self.weights["image_image"] * image_image_output["loss_value"] +
            self.weights["text_text"] * text_text_output["loss_value"]
        )

        return {"loss_value": combined_loss}
## new loss 
class CombinedContrastiveLossImageOnly(nn.Module):
    def __init__(self, model, weights=None):
        super().__init__()
        self.model = model
        self.weights = weights if weights else {"image_text": 1.0, "image_image": 0.5}

        # Initialize sub-losses
        self.image_text_loss = ImageTextContrastiveLoss(model)
        self.image_image_loss = ImageToImageContrastiveLoss(model)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        **kwargs,
    ):
        # Compute individual losses
        image_text_output = self.image_text_loss(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            img_labels=img_labels,
            text_labels=text_labels,
        )

        image_image_output = self.image_image_loss(
            pixel_values=pixel_values,
            img_labels=img_labels,
        )

        

        # Combine the losses using weights
        combined_loss_Ionly = (
            self.weights["image_text"] * image_text_output["loss_value"] +
            self.weights["image_image"] * image_image_output["loss_value"] 
        )

        return {"loss_value": combined_loss_Ionly}
    
class CombinedContrastiveLossTextOnly(nn.Module):
    def __init__(self, model, weights=None):
        super().__init__()
        self.model = model
        self.weights = weights if weights else {"image_text": 1.0, "text_text": 0.5}

        # Initialize sub-losses
        self.image_text_loss = ImageTextContrastiveLoss(model)
        self.text_text_loss = TextToTextContrastiveLoss(model)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        **kwargs,
    ):
        # Compute individual losses
        image_text_output = self.image_text_loss(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            img_labels=img_labels,
            text_labels=text_labels,
        )

        

        text_text_output = self.text_text_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_labels=text_labels,
        )

        # Combine the losses using weights
        combined_lossTO = (
            self.weights["image_text"] * image_text_output["loss_value"] +
            self.weights["text_text"] * text_text_output["loss_value"]
        )

        return {"loss_value": combined_lossTO}