
<div align="center">
<h1>Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance</h1>


<!-- Badges -->
<a><img 
     src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge&logo=data:image/png;base64,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" 
     alt="Github"></a> 
<a href="https://arxiv.org/abs/2512.23461"><img src="https://img.shields.io/badge/arXiv-2512.23461-b31b1b.svg?style=for-the-badge" alt="arXiv"></a> 
<a href="https://github.com/Qwen-Applications/DIR"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a> 
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>


<p align="center">
  <i><b> <img src="https://img.alicdn.com/imgextra/i2/O1CN01FPcQDy1WTPjPX6IH9_!!6000000002789-2-tps-96-96.png" width="16px"  style="vertical-align: middle;"> Qwen Large Model Application Team, Alibaba</b></i>
</p>

In this work, we introduce **Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance**, a novel framework to mitigate format biases (e.g., length, lists, bolding) in reward models (RMs) for large language models. Our approach minimizes the mutual information between the *difference* in response representations and the *relative* bias attributes. This is achieved by training a variational network adversarially against the RM's encoder, encouraging it to learn representations that are invariant to spurious format correlations while retaining true preference signals.

</div>

## ⚙️ 1. Setup and Installation

First, we recommend creating a Conda virtual environment and installing the required dependencies.

```bash
# Create and activate the conda environment
conda create -n dir python=3.9
conda activate dir

# Install all other dependencies
pip install -r requirements.txt
```

## 📥 2. Data and Model Preparation

We provide convenient scripts to download all the necessary datasets (e.g., Skywork-preference-70K-v0.2, bias evaluation sets) and the base models (e.g., Llama-3-8B-Instruct) used in our experiments.

Run the following commands from the project's root directory:

```bash
# Navigate to the scripts directory
cd scripts

# Download all required datasets
bash auto_download_data.sh

# Download the base language models
bash auto_download_model.sh
```

After the scripts complete, your data and models will be organized in the designated directories.

## 🚀 3. Training and Evaluation Pipeline

The full experimental pipeline consists of three main stages: training the debiased reward model, aligning a policy model using PPO, and evaluating the final policy.

### Step 3.1: Train the Debiased Reward Model (DIR)

To train our debiased reward model using the DIR framework, run the `train_debias_rm.sh` script. This script orchestrates the training process defined in `reward_models/run_debias_reward_models_train.py`.

```bash
# Make sure you are in the scripts/ directory
bash train_debias_rm.sh
```

The training logs and final RM checkpoints will be saved to the output directory specified within the script (e.g., `../exp/debiased_rm`).

### Step 3.2: Align a Policy with PPO

Once the debiased RM is trained, we use it to provide rewards for aligning a policy model with Proximal Policy Optimization (PPO) using the `ms-swift` evaluation tool.

**Important:** Before running, you must edit `ms_ppo_script.sh` and update the `REWARD_MODEL_PATH` variable to point to the checkpoint of the debiased RM you trained in the previous step. Plase make sure you have cloned the MS-Swift successfully.

```bash
# Example modification inside ms_ppo_script.sh:
# REWARD_MODEL_PATH="../exp/debiased_rm/checkpoint-final"

# Run the PPO training script
bash ms_ppo_script.sh
```

This will train a policy model and save the checkpoints to the specified output directory.

### Step 3.3: Evaluate the Final Aligned Policy

Finally, we evaluate the performance of the PPO-aligned policy model on various benchmarks using the `evalscope` evaluation tool.

**Important:** Before running, you must edit `rm_eval/evalscope_evaluation_script.sh` and update the `MODEL_PATH` variable to point to the PPO-aligned model checkpoint from Step 3.2. Plase make sure you have cloned the EvalScope successfully.

```bash
# From the root directory, run the evaluation script
bash rm_eval/evalscope_evaluation_script.sh
```

The script will generate responses for the benchmark prompts and compute the final evaluation scores, saving the results to the specified output directory.

## 📁 Repository Structure

```
.
├── deepspeed_configs/     # DeepSpeed configuration files
├── reward_models/         # Core logic for training all reward models
│   ├── debias_trainer.py  # Trainer implementing the DIR framework
│   └── run_debias_reward_models_train.py # Main script to launch DIR training
├── rm_eval/               # Scripts for evaluating reward and policy models
│   ├── eval_biasbench.py  # Evaluate format bias
│   └── evalscope_evaluation_script.sh # Evaluate policy performance
├── scripts/               # Main workflow orchestration scripts
│   ├── auto_download_data.sh
│   ├── auto_download_model.sh
│   ├── train_debias_rm.sh # Use this to train our model
│   └── ms_ppo_script.sh   # Use this for PPO alignment
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## 🙏 Acknowledgements

This project is built upon several fantastic open-source libraries. We would like to extend our heartfelt gratitude to the developers and communities of:
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for providing easy access to state-of-the-art models.
- [Hugging Face TRL](https://github.com/huggingface/trl) for the robust `RewardTrainer` which served as the foundation for our DebiasTrainer.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for enabling efficient large-model training.
- [ModelScope Swift](https://github.com/modelscope/ms-swift) and [EvalScope](https://github.com/modelscope/evalscope) for the powerful PPO and evaluation frameworks.
- The [Generalizable-Reward-Model](https://github.com/YangRui2015/Generalizable-Reward-Model) repository for providing a useful data processing script.

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{li2025eliminatinginductivebiasreward,
      title={Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance}, 
      author={Zhuo Li and Pengyu Cheng and Zhechao Yu and Feifei Tong and Anningzhe Gao and Tsung-Hui Chang and Xiang Wan and Erchao Zhao and Xiaoxi Jiang and Guanjun Jiang},
      year={2025},
      eprint={2512.23461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.23461}, 
}
```
