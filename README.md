# RAPO-ASAG



<img src=".\Images\test.png">

# Environment
conda create -n LLMGrading python=3.10

conda activate LLMGrading

pip install -r requirements.txt

# Datasets Download
scientsbank: https://huggingface.co/datasets/nkazi/SciEntsBank

beetle: https://huggingface.co/datasets/nkazi/Beetle

Create a folder named DataSet, Put the dataset in the DataSet directory

# Supervised Warm-Start

python --model_name microsoft/deberta-v3-base --dataset beetle --output_dir results_ce_ppo  --lr 1e-5 --sft_epochs 5

# RAPO

python PAPO.py --model_name microsoft/deberta-v3-base --dataset beetle  --sft_best_dir results_ce_ppo/deberta-v3-base_scientsbank/sft_best -rl_epochs 2 --ppo_epochs 4
--rl_lr 1e-5  --ppo_clip 0.2 --false_negative_reward -1.5 --false_positive_reward -1.2

# Main experimental results
<img src=".\Images\T1.png">

<img src=".\Images\T2.png">

<img src=".\Images\T3.png">
