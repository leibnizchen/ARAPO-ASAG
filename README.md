# RAPO-ASAG: Reward-Aligned Proximal Optimization for Automatic Short Answer Grading

Automatic Short Answer Grading (ASAG) has achieved significant progress with pretrained language models, yet existing methods mainly optimize supervised objectives that treat grading errors equally and fail to capture pedagogical priorities. In this paper, we propose RAPO-ASAG, a Reward-Aligned Proximal Optimization framework that reformulates ASAG as a contextual bandit problem and optimizes grading decisions through educationally aligned rewards. RAPO-ASAG first obtains a reliable grading policy through supervised warm-start training. It then performs constrained policy refinement using a PPO-style clipped objective and an asymmetric utility function that assigns different penalties to false negatives and false positives. Rather than using PPO for sequential credit assignment, we use clipping, supervised anchoring, KL regularization, and entropy regularization to limit policy drift during reward-based refinement. Experiments on SciEntsBank and Beetle show that RAPO-ASAG consistently outperforms strong baselines, achieving Macro-F1 scores of 0.8278 on SciEntsBank unseen-answer evaluation and 0.9083 on Beetle unseen-answer evaluation. Further analysis reveals improved calibration and fewer critical grading errors, validating the effectiveness of our reward-aligned framework for reliable assessment.

<div align=center> 
<img src=".\Image\F1.png">
</div>
 
# Environment
conda create -n LLMGrading python=3.10

conda activate LLMGrading

pip install -r requirements.txt

# Datasets Download
scientsbank: https://huggingface.co/datasets/nkazi/SciEntsBank

beetle: https://huggingface.co/datasets/nkazi/Beetle

Create a folder named DataSet, Put the dataset in the DataSet directory

# Supervised Warm-Start
```
python stf.py \
  --model_name microsoft/deberta-v3-base \
  --dataset beetle \
  --output_dir results_ce_ppo \
  --lr 1e-5 \
  --sft_epochs 5
```
# RAPO
```
python PAPO.py \
  --model_name microsoft/deberta-v3-base \ 
  --dataset beetle \
  --sft_best_dir results_ce_ppo/deberta-v3-base_scientsbank/sft_best \
  --rl_epochs 2 \
  --ppo_epochs 4 \
  --rl_lr 1e-5 \
  --ppo_clip 0.2 \
  --false_negative_reward -1.5 \
  --false_positive_reward -1.2
```

# Main experimental results
<img src=".\Image\T1.png">

<img src=".\Image\T2.png">

<img src=".\Image\F2.png">

<img src=".\Image\T3.png">

<img src=".\Image\T4.png">

<img src=".\Image\T5.png">
