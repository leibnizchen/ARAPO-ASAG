# RAPO-ASAG

Large language models (LLMs) have shown strong performance in automated educational assessment. However, recent studies reveal that student responses may contain prompt-cheating instructions that manipulate grading outcomes, threatening the reliability of LLM-based assessment systems. To address this issue, we formulate robust answer scoring under prompt cheating as a selective prediction problem. We propose a coverage-constrained framework that jointly performs answer scoring and cheating detection. The model predicts both a grading score and a rejection probability, allowing suspicious responses to be selectively rejected. A coverage constraint is introduced to prevent trivial rejection strategies, while a cheating-aware supervision objective improves attack detection. Experiments on SciEntsBank and Beetle augmented with five categories of prompt-cheating attacks show that our method consistently outperforms confidence-based rejection, detector-grader pipelines, and multi-task learning baselines. Our method achieved average cheat detection accuracy of 97.07\% and 98.28\% on the two datasets, respectively. Moreover, grading accuracy on clean answer is maintained or improved relative to competing approaches. The proposed method providing a practical solution for secure and reliable LLM-based educational assessment.

<img src=".\Images\test.png">

# Environment
conda create -n LLMGrading python=3.10

conda activate LLMGrading

pip install -r requirements.txt

# Datasets Download
scientsbank: https://huggingface.co/datasets/nkazi/SciEntsBank

beetle: https://huggingface.co/datasets/nkazi/Beetle

Create a folder named DataSet, Put the dataset in the DataSet directory

# Main train
python main.py  ----model_name [model name] --dataset [dataset] --lr [learning rate] --cheat_ratio [trainset cheating rate]  --eval_attack_ratio [testset cheating rate]  --reject_threshold [reject grading threshold]
## example
python main.py  ----model_name /Qwen/Qwen3-4B --dataset scientsbank --lr 2e-5 --cheat_ratio 0.1  --eval_attack_ratio 0.1  --reject_threshold 0.5


# baseline
python baselines_asag_cheat.py  ----model_name /Qwen/Qwen3-4B --dataset scientsbank --lr 2e-5 --cheat_ratio 0.1  --eval_attack_ratio 0.1  --reject_threshold 0.5 --baseline vanilla 

choices=["vanilla", "confidence", "detector_pipeline", "multitask"]


**Vanilla Scorer:** A standard short-answer scoring model with the same backbone and input format as our method, trained solely for binary scoring.

**Vanilla Scorer with Confidence Rejection:** The vanilla scorer augmented with a confidence-based rejection mechanism. Predictions with maximum softmax probability below a predefined threshold are rejected. 

**External Detector and Scorer:** A pipeline approach consisting of an independent attack detector and scoring model. Inputs identified as attacks are rejected; otherwise, the grader produces the final score. 

**Multi-task Scorer:** A joint model with scoring and attack-detection heads trained using a standard multi-task objective. Unlike our method, it does not explicitly couple rejection decisions with scoring risk. 


# Main experimental results
<img src=".\Images\T1.png">

<img src=".\Images\T2.png">

<img src=".\Images\T3.png">
