from accelerate import Accelerator
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return {
            "accuracy": float(
                accuracy_score(labels, predictions)
            )
        }

def print_trainable_parameters(model, print_trainable_name=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_trainable_name:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


