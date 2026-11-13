# ARAPO-ASAG: Asymmetric Reward-Aligned Proximal Optimization for Automatic Short Answer Grading

Automatic Short Answer Grading (ASAG) has advanced with pretrained language models, yet conventional supervised objectives do not explicitly account for the unequal educational costs of grading errors. In settings where rejecting a correct answer is more costly than accepting an incorrect one, this mismatch motivates cost-aware optimization. We propose RAPO-ASAG, a Reward-Aligned Proximal Optimization framework that combines asymmetric utility with constrained policy refinement. Formulating grading as a contextual bandit problem, RAPO-ASAG first establishes a reliable policy through supervised warm-start training and then refines it using a PPO-style clipped objective. An asymmetric reward penalizes false negatives more heavily than false positives, while supervised anchoring, KL regularization, and entropy regularization promote stable policy refinement. Experiments on SciEntsBank and Beetle show consistent improvements over the evaluated supervised and reinforcement learning baselines, achieving Macro-F1 scores of 0.8278 and 0.9083 on their respective unseen-answer evaluations. Further analyses demonstrate improved calibration in most settings and support the contributions of asymmetric rewards and stabilization mechanisms to predictive performance.

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
