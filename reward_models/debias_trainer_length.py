"""
Debias Reward Model Trainer
Supports debias network and adaptive margin for reward model training
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple
from accelerate import Accelerator
import numpy as np

from base_trainer import RewardTrainer
from transformers import Trainer, TrainingArguments
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.trainer_utils import PredictionOutput

from dataclasses import dataclass, field
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from accelerate.utils import (
    DistributedType,
)

def post_process_RBv1(rewards_j, rewards_k, metric_key_prefix):
    acc = (rewards_j > rewards_k).astype(np.float16).mean()
    reward_diff = (rewards_j - rewards_k).astype(np.float16).mean()
    metrics = {
        f"{metric_key_prefix}_RBv1_accuracy": acc.item(),
        f"{metric_key_prefix}_RBv1_avg_rewards_chosen": rewards_j.mean().item(),
        f"{metric_key_prefix}_RBv1_avg_rewards_rejected": rewards_k.mean().item(),
        f"{metric_key_prefix}_RBv1_avg_margin": reward_diff.item(),
    }
    return metrics

def post_process_RMBench(
    expanded_domains: List[str], 
    domains_to_analyze: List[str], 
    scores: np.ndarray,
    metric_key_prefix: str
) -> Dict[str, float]:

    
    def compute_h_n_e_accuracy(scores_subset: np.ndarray) -> Dict[str, float]:
        MATRIX_SIZE = 3
        num_rows = scores_subset.shape[0]

        assert num_rows % MATRIX_SIZE == 0
        
        num_samples = num_rows // MATRIX_SIZE
        scores_reshaped = scores_subset.reshape(num_samples, MATRIX_SIZE, 2)
        chosen_scores = scores_reshaped[:, :, 0]
        rejected_scores = scores_reshaped[:, :, 1]
        
        chosen_exp = np.expand_dims(chosen_scores, axis=2)
        rejected_exp = np.expand_dims(rejected_scores, axis=1)
        
        victory_matrices = chosen_exp > rejected_exp
        acc_matrix = victory_matrices.astype(float).mean(axis=0)

        upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
        
        normal_acc = np.mean(np.diag(acc_matrix))
        
        lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count
        
        return {"hard_acc": hard_acc, "normal_acc": normal_acc, "easy_acc": easy_acc}

    expanded_domains = np.array(expanded_domains)
    
    domain_results = {}
    for domain in domains_to_analyze:
        mask = np.char.startswith(expanded_domains, domain)
        domain_results[domain] = compute_h_n_e_accuracy(scores[mask])

    domain_avg_results = {
        domain: np.mean(list(metrics.values()))
        for domain, metrics in domain_results.items()
        if not np.isnan(list(metrics.values())[0])
    }
    
    all_hard = [m['hard_acc'] for m in domain_results.values() if not np.isnan(m['hard_acc'])]
    all_normal = [m['normal_acc'] for m in domain_results.values() if not np.isnan(m['normal_acc'])]
    all_easy = [m['easy_acc'] for m in domain_results.values() if not np.isnan(m['easy_acc'])]
    
    domain_h_n_e_acc = {
        "hard_acc": np.mean(all_hard) if all_hard else np.nan,
        "normal_acc": np.mean(all_normal) if all_normal else np.nan,
        "easy_acc": np.mean(all_easy) if all_easy else np.nan,
    }

    all_avg = list(domain_avg_results.values())
    total_avg_acc = np.mean(all_avg) if all_avg else np.nan

    final_results = {}
    final_results.update(domain_avg_results)
    final_results.update(domain_h_n_e_acc)
    final_results["total_avg_acc"] = total_avg_acc
    
    metrics = {
        f"{metric_key_prefix}_RMBench_Chat": final_results['chat'],
        f"{metric_key_prefix}_RMBench_Math": final_results['math'],
        f"{metric_key_prefix}_RMBench_Code": final_results['code'],
        f"{metric_key_prefix}_RMBench_Safety": final_results['safety'],
        f"{metric_key_prefix}_RMBench_Hard": final_results['hard_acc'],
        f"{metric_key_prefix}_RMBench_Normal": final_results['normal_acc'],
        f"{metric_key_prefix}_RMBench_Easy": final_results['easy_acc'],
        f"{metric_key_prefix}_RMBench_total": final_results['total_avg_acc'],
    }

    return metrics


class AllGatherWithGrad(torch.autograd.Function):
    """
    An autograd function that performs all-gather on a tensor and properly
    handles gradients in the backward pass.
    """
    @staticmethod
    def forward(ctx, tensor_to_gather):
        """
        Forward pass: gathers tensors from all GPUs.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor_to_gather

        world_size = dist.get_world_size()
        gathered_tensors = [torch.empty_like(tensor_to_gather) for _ in range(world_size)]
        
        dist.all_gather(gathered_tensors, tensor_to_gather)
        output = torch.cat(gathered_tensors, dim=0)
        
        ctx.world_size = world_size
        ctx.input_shape = tensor_to_gather.shape
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: scatters gradients back to each GPU.
        """
        if not dist.is_available() or not dist.is_initialized():
            return grad_output

        grad_output_reshaped = grad_output.view(ctx.world_size, *ctx.input_shape)
        rank = dist.get_rank()
        grad_input = grad_output_reshaped[rank]

        return grad_input


class CLUBForCategorical(nn.Module):
    """CLUB estimator for categorical labels
    
    This class provides a CLUB estimator to calculate MI upper bound between 
    vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    """
    
    def __init__(self, input_dim, label_num, hidden_size=None):
        """
        Args:
            input_dim: the dimension of input embeddings
            label_num: the number of categorical labels 
            hidden_size: hidden size for the variational network
        """
        super().__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        """
        Args:
            inputs: shape [batch_size, input_dim], a batch of embeddings
            labels: shape [batch_size], a batch of label index
        """
        logits = self.variational_net(inputs)  # [sample_size, label_num]
        
        sample_size, label_num = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - F.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        """Log-likelihood"""
        logits = self.variational_net(inputs)
        return -F.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        """Learning loss"""
        return -self.loglikeli(inputs, labels)


class DeBiasedRewardTrainer(RewardTrainer): 
    def __init__(self, **kwargs):
        self.use_debias = kwargs.pop("use_debias")
        self.debias_type = kwargs.pop("debias_type")
        self.debias_lr = kwargs.pop("debias_lr")
        self.debias_hidden_dim = kwargs.pop("debias_hidden_dim")
        self.debias_factor = kwargs.pop("debias_factor")
        self.debias_optim_iter = kwargs.pop("debias_optim_iter")
        self.debias_projection = kwargs.pop("debias_projection")
        super().__init__(**kwargs)

        model_hidden_size = getattr(self.model.config, "hidden_size", 4096)

        if self.debias_type == "concat":
            input_dim = model_hidden_size * 2
        else: # "difference"
            input_dim = model_hidden_size
        
        self.debias_network = CLUBForCategorical(
            input_dim=input_dim,
            label_num=2, 
            hidden_size=self.debias_hidden_dim
        )
        self.debias_network.to(self.accelerator.device)

        self.debias_optimizer = torch.optim.AdamW(
            self.debias_network.parameters(),
            lr=self.debias_lr  
        )
        
        self.log_info = {
            "loss_preference": [],
            "loss_debias_net": [],
            "loss_mi": []
            }
            
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
            _dataset_name = list(set(self.eval_dataset['dataset']))[0]
            if _dataset_name == 'RBv1':
                metrics = post_process_RBv1(rewards_j, rewards_k, metric_key_prefix)
            elif _dataset_name == 'RMBench':
                domains = ["chat", "math", "code", "safety"]
                metrics = post_process_RMBench(self.eval_dataset['domain'], domains, rewards, metric_key_prefix)
            elif _dataset_name == 'RBv2':
                metrics = post_process_RBv1(rewards_j, rewards_k, metric_key_prefix)
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

        loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() 

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

    def train_debias_nn(self, h_with_grad, A_labels):

        h_no_grad = h_with_grad.detach()

        h_chosen_no_grad, h_rejected_no_grad = h_no_grad[0::2], h_no_grad[1::2]
        local_X_no_grad = (h_chosen_no_grad - h_rejected_no_grad) if self.debias_type == "difference" else torch.cat([h_chosen_no_grad, h_rejected_no_grad], dim=-1)

        with torch.no_grad():
            global_X = self.accelerator.gather(local_X_no_grad)
            global_Y = self.accelerator.gather(A_labels)

        self.debias_network.train()
        loss_debias_net_val = 0.0
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for _ in range(self.debias_optim_iter):
                self.debias_optimizer.zero_grad()
                loss = self.debias_network.learning_loss(global_X, global_Y)
                loss.backward()
                self.debias_optimizer.step()
            loss_debias_net_val = loss.detach().item()

        self.debias_network.eval()

        return loss_debias_net_val

    def estimate_mi(self, h_with_grad, A_labels):

        h_chosen_grad, h_rejected_grad = h_with_grad[0::2], h_with_grad[1::2]
        X_for_debias = (h_chosen_grad - h_rejected_grad) if self.debias_type == "difference" else torch.cat([h_chosen_grad, h_rejected_grad], dim=-1)
            
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            estimated_mi = self.debias_network(X_for_debias, A_labels)
    
        return estimated_mi

    def global_estimate_mi(self, h_with_grad, A_labels):
        h_chosen_grad, h_rejected_grad = h_with_grad[0::2], h_with_grad[1::2]
        X_for_debias = (h_chosen_grad - h_rejected_grad) if self.debias_type == "difference" else torch.cat([h_chosen_grad, h_rejected_grad], dim=-1)

        global_X_for_debias = AllGatherWithGrad.apply(X_for_debias)
        global_Y = self.accelerator.gather(A_labels)
            
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            global_estimated_mi = self.debias_network(global_X_for_debias, global_Y)
    
        return global_estimated_mi

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

    # before
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        with self.compute_loss_context_manager():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
            rewards = outputs.logits
            last_hidden_state_with_grad = outputs.hidden_states[-1]
            sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
            
            rewards_j, rewards_k = rewards[0::2], rewards[1::2]
            h_with_grad = last_hidden_state_with_grad[torch.arange(last_hidden_state_with_grad.shape[0]), sequence_lengths]
            len_chosen, len_rejected = sequence_lengths[0::2], sequence_lengths[1::2]
            A_labels = (len_chosen > len_rejected).long()

            loss_preference = -F.logsigmoid(rewards_j - rewards_k).mean()
            loss_debias_net_val = self.train_debias_nn(h_with_grad, A_labels)
            # estimated_mi = self.estimate_mi(h_with_grad, A_labels)
            global_mi = self.global_estimate_mi(h_with_grad, A_labels)

            loss = loss_preference + self.debias_factor * global_mi

        if self.state.is_world_process_zero and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_preference": loss_preference.item(),
                "loss_debias_net": loss_debias_net_val,
                "loss_mi": global_mi.item(),
            })

        del inputs

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()