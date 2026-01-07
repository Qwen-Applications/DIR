# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import RunningMoments, selective_log_softmax

from swift.utils import get_logger
from ..mixin import DataLoaderMixin, SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__
logger = get_logger()

import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple

import torch.distributed as dist

# 添加这个类
class AllGatherWithGrad(torch.autograd.Function):
    """All-gather with gradient propagation"""
    @staticmethod
    def forward(ctx, tensor_to_gather):
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
        if not dist.is_available() or not dist.is_initialized():
            return grad_output
        grad_output_reshaped = grad_output.view(ctx.world_size, *ctx.input_shape)
        rank = dist.get_rank()
        return grad_output_reshaped[rank]

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

class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):
    def __init__(self,
                model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                *_args,
                **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        if 'loss_weights' in DPOConfig.__dict__:
            # trl >= 0.20
            self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
            self.loss_weights = args.loss_weights
        else:
            self.loss_type = args.loss_type

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for loss_type in loss_types:
            if (loss_type in ['hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'apo_zero', 'apo_down']
                    and args.label_smoothing > 0):
                warnings.warn(
                    f'You are using the {loss_type} loss type that does not support label smoothing. The '
                    '`label_smoothing` parameter will be ignored. '
                    'Set `label_smoothing` to `0.0` to remove this warning.',
                    UserWarning,
                )
            if loss_type == 'kto_pair':
                raise ValueError('Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.')

        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free
        self.use_weighting = False
        
        # Initialize debias configuration BEFORE super().__init__
        self.use_debias = getattr(args, 'use_debias', False)
        if self.use_debias or 'dpo_debias' in loss_types:
            self.use_debias = True
            self.debias_type = getattr(args, 'debias_type', 'difference')
            self.debias_factor = getattr(args, 'debias_factor', 1)
            self.debias_lr = getattr(args, 'debias_lr', 1e-4)
            self.debias_hidden_dim = getattr(args, 'debias_hidden_dim', 1024)
            self.debias_optim_iter = getattr(args, 'debias_optim_iter', 5)
            self.debias_warmup_steps = 80  

        super().__init__(model, ref_model, *_args, **kwargs)

        # Initialize components that require self.accelerator AFTER super().__init__
        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)
        
        if self.use_debias:
            model_hidden_size = getattr(self.model.config, "hidden_size", 4096)
            input_dim = model_hidden_size * 2 if self.debias_type == "concat" else model_hidden_size
            
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
            
            logger.info(f"Debias initialized: type={self.debias_type}, factor={self.debias_factor}")

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
    
        batch = batch.copy()
        labels = batch.pop('labels', None)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            labels, logits_to_keep = self.get_logits_to_keep(labels)
            if logits_to_keep is not None:
                batch['logits_to_keep'] = logits_to_keep
        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        if self.is_encoder_decoder:
            batch['labels'] = labels
        position_ids = batch.pop('_position_ids', None)
        if position_ids is None:
            position_ids = batch.get('position_ids')
        outputs = model(**batch, use_cache=False, output_hidden_states=return_hidden_states)
        all_logits = outputs.logits

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder and self.template.sequence_parallel_size == 1:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, shifts=-1, dims=1)
        per_token_logps, mean_all_logits, loss_mask = self.get_per_token_logps(
            all_logits, labels, label_pad_token_id=self.label_pad_token_id)
        origin_per_token_logps = per_token_logps

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if 'ipo' in loss_types:
            size_completion = loss_mask.sum(dim=-1)
            per_token_logps = per_token_logps / size_completion

        output = {}
        if self.template.padding_free:
            cu_seqlens = self.get_cu_seqlens(position_ids, batch.get('logits_to_keep'))
            all_logps = per_token_logps.new_zeros((cu_seqlens.shape[0] - 1, ))
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                all_logps[i] = per_token_logps[:, start:end].sum()
            num_examples = all_logps.shape[0] // 2
            num_tokens = cu_seqlens[num_examples]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
            output['chosen_lengths'] = torch.tensor([
                loss_mask[:, cu_seqlens[i]:cu_seqlens[i + 1]].sum().item()
                for i in range(num_examples)
            ], device=all_logps.device, dtype=torch.float)
            output['rejected_lengths'] = torch.tensor([
                loss_mask[:, cu_seqlens[num_examples + i]:cu_seqlens[num_examples + i + 1]].sum().item()
                for i in range(num_examples)
            ], device=all_logps.device, dtype=torch.float) 
            if return_hidden_states and hasattr(outputs, 'hidden_states'):
                last_hidden_states = outputs.hidden_states[-1]
                h_list = []
                for i in range(cu_seqlens.shape[0] - 1):
                    start, end = cu_seqlens[i], cu_seqlens[i + 1]
                    h_list.append(last_hidden_states[0, end - 1, :])
                output['hidden_states'] = torch.stack(h_list)

        else:
            all_logps = per_token_logps.sum(-1)
            num_examples = labels.shape[0] // 2
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:num_examples][loss_mask[:num_examples]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:num_examples][loss_mask[:num_examples]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_examples:][loss_mask[num_examples:]].mean()
            output['chosen_lengths'] = loss_mask[:num_examples].sum(dim=-1).float()
            output['rejected_lengths'] = loss_mask[num_examples:].sum(dim=-1).float()
            if return_hidden_states and hasattr(outputs, 'hidden_states'):
                last_hidden_states = outputs.hidden_states[-1]
                batch_size = last_hidden_states.shape[0]
                sequence_lengths = loss_mask.sum(dim=-1) - 1
                output['hidden_states'] = last_hidden_states[torch.arange(batch_size), sequence_lengths]
                
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    @staticmethod
    def get_per_token_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id=-100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        loss_mask = labels != label_pad_token_id
        labels = labels.clone()
        labels[~loss_mask] = 0
        # https://github.com/huggingface/trl/pull/2799
        # Reduce peak vram consumption with efficient selective log_softmax
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        return per_token_logps, logits.mean(-1), loss_mask

    def training_step(self, model, inputs, *args, **kwargs):
        inputs['_position_ids'] = inputs.get('position_ids')
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        inputs['_position_ids'] = inputs.get('position_ids')
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)

    def train_debias_nn(self, h_chosen, h_rejected, A_labels):
        """Train debias network"""
        h_chosen_no_grad = h_chosen.detach()
        h_rejected_no_grad = h_rejected.detach()
        
        if self.debias_type == "difference":
            local_X = h_chosen_no_grad - h_rejected_no_grad
        else:
            local_X = torch.cat([h_chosen_no_grad, h_rejected_no_grad], dim=-1)
        
        with torch.no_grad():
            global_X = self.accelerator.gather(local_X)
            global_Y = self.accelerator.gather(A_labels)
        
        self.debias_network.train()
        for _ in range(self.debias_optim_iter):
            self.debias_optimizer.zero_grad()
            loss = self.debias_network.learning_loss(global_X, global_Y)
            loss.backward()
            self.debias_optimizer.step()
        
        self.debias_network.eval()
        
        return loss.detach().item()

    def global_estimate_mi(self, h_chosen, h_rejected, A_labels):
        """Estimate MI with gradient"""
        if self.debias_type == "difference":
            X_for_debias = h_chosen - h_rejected
        else:
            X_for_debias = torch.cat([h_chosen, h_rejected], dim=-1)
        
        global_X = AllGatherWithGrad.apply(X_for_debias)
        global_Y = self.accelerator.gather(A_labels)
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            mi = self.debias_network(global_X, global_Y)
            
        return mi

    def get_batch_loss_metrics(self, model, batch: Dict[str, Union[List, torch.LongTensor]], train_eval: str = "train"):
        """Override to add debias loss"""
        metrics = {}
        
        # 修改这里：eval 时也提取 hidden states
        forward_output = self.concatenated_forward(
            model, batch, is_ref_model=False,
            return_hidden_states=self.use_debias 
        )
        
        policy_chosen_logps = forward_output["chosen_logps"]
        policy_rejected_logps = forward_output["rejected_logps"]
        chosen_lengths = forward_output.get("chosen_lengths")
        rejected_lengths = forward_output.get("rejected_lengths")

        if ("reference_chosen_logps" in batch and "reference_rejected_logps" in batch 
            and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
                else:
                    reference_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
            reference_chosen_logps = reference_output["chosen_logps"]
            reference_rejected_logps = reference_output["rejected_logps"]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )

        prefix = "eval_" if train_eval == "eval" else ""

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        if self.args.rpo_alpha is not None:
            losses = losses * torch.exp(policy_rejected_logps * self.args.rpo_alpha)

        # 修改这里：区分 train 和 eval 的处理
        if self.use_debias and 'hidden_states' in forward_output:

            if train_eval == "train":
                losses = losses

            h_with_grad = forward_output['hidden_states']
            num_pairs = h_with_grad.shape[0] // 2
            h_chosen = h_with_grad[:num_pairs]
            h_rejected = h_with_grad[num_pairs:]
            A_labels = (chosen_lengths > rejected_lengths).long()
            
            if train_eval == "train" and self.state.global_step > self.debias_warmup_steps:
                loss_debias_net = self.train_debias_nn(h_chosen, h_rejected, A_labels)
                mi = self.global_estimate_mi(h_chosen, h_rejected, A_labels)
                
                losses = losses + self.debias_factor * mi
                
                metrics[f"{prefix}debias/net_loss"] = float(loss_debias_net)
                metrics[f"{prefix}debias/mi_estimate"] = float(mi.item())
                metrics[f"{prefix}debias/regularization"] = float((self.debias_factor * mi).item())

            else:
                with torch.no_grad():
                    mi = self.global_estimate_mi(h_chosen, h_rejected, A_labels)
                
                metrics[f"{prefix}debias/mi_estimate"] = float(mi.item())

        if self.loss_type == "dpo_lc":
            device = self.accelerator.device
            logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            # DPO with Length-Controlled regularization
            # Formula: -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) + (α|y_w| - α|y_l|))]
            
            # Get the length regularization coefficient alpha
            alpha = 0.01  # Default value, adjust as needed
            if hasattr(self, 'length_regularization_alpha'):
                alpha = self.length_regularization_alpha
            elif self.f_divergence_params and 'length_regularization_alpha' in self.f_divergence_params:
                alpha = float(self.f_divergence_params['length_regularization_alpha'])
            
            # Get lengths from the concatenated forward output
            chosen_lengths = chosen_lengths.to(device)  # These should be passed from concatenated_forward
            rejected_lengths = rejected_lengths.to(device)
            
            # Calculate the length difference term: α(|y_w| - |y_l|)
            length_diff = alpha * (chosen_lengths - rejected_lengths)
            
            # Add the length regularization term to the logits
            regularized_logits = self.beta * logits + length_diff
            
            # Apply the sigmoid loss with the regularized logits
            losses = (
                -F.logsigmoid(regularized_logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-regularized_logits) * self.label_smoothing
            )

        # 标准 metrics
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = forward_output["mean_rejected_logits"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = forward_output["mean_chosen_logits"].detach().mean().cpu()
        
        if chosen_lengths is not None:
            metrics[f"{prefix}lengths/chosen"] = chosen_lengths.mean().cpu()
            metrics[f"{prefix}lengths/rejected"] = rejected_lengths.mean().cpu()

        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = forward_output["aux_loss"].detach().mean().cpu()

        return losses.mean(), metrics
