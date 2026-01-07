from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.trainer_utils import PredictionOutput


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        bias_labels = []
        
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )

            if 'margin' in feature.keys():
                margins.append(feature['margin'])
            if 'bias_label' in feature.keys():
                bias_labels.append(feature['bias_label'])

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "margin": margins,
            "bias_label": bias_labels
        }
        return batch


class SimpleRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'bt')
        self.weight_ratio = kwargs.pop('weight_ratio', 0.1)
        super(SimpleRewardTrainer, self).__init__(**kwargs)
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
    # 对于RewardBench-V2，可以在这里重写
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        return output

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str = "Evaluating",
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        rewards = output.predictions
        rewards_j = output.predictions[:,0]  # 已经是gather好的
        rewards_k = output.predictions[:,1]  
        
        if self.accelerator.is_main_process:
            acc = (rewards_j > rewards_k).astype(np.float16).mean()
            reward_diff = (rewards_j - rewards_k).astype(np.float16).mean()
            metrics = {
                f"{metric_key_prefix}_accuracy": acc.item(),
                f"{metric_key_prefix}_avg_rewards_chosen": rewards_j.mean().item(),
                f"{metric_key_prefix}_avg_rewards_rejected": rewards_k.mean().item(),
                f"{metric_key_prefix}_avg_margin": reward_diff.item(),
            }
        else:
            metrics = {}

        output.metrics.update(metrics)

        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        if self.loss_type == 'bt':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() 
        elif self.loss_type == 'pos_reg':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_j.mean())
        elif self.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
        elif self.loss_type == 'labelsmooth':
            loss = - (1-self.weight_ratio) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        else:
            raise NotImplementedError

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss