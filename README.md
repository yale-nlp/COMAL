# COMAL

This repository is for our paper "COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences".

## Outline

- [How to Run](#how-to-run)
    - [Installation and Requirements](#installation-and-requirements)
    - [Running COMAL](#running-comal)
- [Model Checkpoints](#model-checkpoints)
    - [Best Checkpoints](#best-checkpoints)
    - [All Checkpoints](#all-checkpoints)
- [File Structure](#file-structure)
    - [Files](#files)
    - [Directories](#directories)

## How to Run

### Installation and Requirements
Please run `pip install -r requirements.txt` to install the required packages.

To run COMAL, you will likely need at least 4 GPUs with 48GB of memory each. The code is tested on a machine with 8 NVIDIA A6000 Ada GPUs.

### Running COMAL
To run COMAL, please use the following command: `bash comal.sh`.

The script [`comal.sh`](comal.sh) performs iterative preference optimization with the following steps:
1. Sampling candidate outputs from the LLM.
2. Scoring the candidate outputs using the preference model.
3. Data processing and precomputing the log probabilities of the output pairs.
4. Training: updating the LLM using INPO.
5. Evaluating the LLM.

## Model Checkpoints

Below are the model checkpoints we trained with different iterative preference optimization methods.

**Iter-IPO** is the iterative IPO method, **INPO-Small** is the INPO method with a small regularization coefficient, **INPO-Large** is the INPO method with a large regularization coefficient, and **COMAL** is the COMAL method.

### Best Checkpoints
The best checkpoints produced by different algorithms are provided below, with the corresponding training round.

| Method | Checkpoint | Round |
|--------|------------|-------|
| COMAL | [yale-nlp/comal-qwen2-1.5b](yale-nlp/comal-qwen2-1.5b) | 7 |
| INPO-Small | [yale-nlp/comal-qwen2-1.5b-inpo-small](yale-nlp/comal-qwen2-1.5b-inpo-small) | 5 |
| INPO-Large | [yale-nlp/comal-qwen2-1.5b-inpo-large](yale-nlp/comal-qwen2-1.5b-inpo-large) | 4 |


### All Checkpoints
The checkpoints produced at the end of each training round are provided below.
Each training round consists 6 training iterations.
Please refer to the paper for more details.

| Round (Iteration) | Iter-IPO | INPO-Small | INPO-Large | COMAL |
|-------|----------|------------|------------|-------|
| 1 (6)    | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round1](yale-nlp/comal-qwen2-1.5b-iter-ipo-round1) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round1](yale-nlp/comal-qwen2-1.5b-inpo-small-round1) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round1](yale-nlp/comal-qwen2-1.5b-inpo-large-round1) | Same as [INPO-Small-Round1](yale-nlp/comal-qwen2-1.5b-inpo-small-round1) |
| 2 (12)    | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round2](yale-nlp/comal-qwen2-1.5b-iter-ipo-round2) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round2](yale-nlp/comal-qwen2-1.5b-inpo-small-round2) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round2](yale-nlp/comal-qwen2-1.5b-inpo-large-round2) | Same as [INPO-Small-Round2](yale-nlp/comal-qwen2-1.5b-inpo-small-round2) |
| 3 (18)    | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round3](yale-nlp/comal-qwen2-1.5b-iter-ipo-round3) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round3](yale-nlp/comal-qwen2-1.5b-inpo-small-round3) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round3](yale-nlp/comal-qwen2-1.5b-inpo-large-round3) | [yale-nlp/comal-qwen2-1.5b-round3](yale-nlp/comal-qwen2-1.5b-round3) |
| 4 (24)   | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round4](yale-nlp/comal-qwen2-1.5b-iter-ipo-round4) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round4](yale-nlp/comal-qwen2-1.5b-inpo-small-round4) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round4](yale-nlp/comal-qwen2-1.5b-inpo-large-round4) | [yale-nlp/comal-qwen2-1.5b-round4](yale-nlp/comal-qwen2-1.5b-round4) |
| 5 (30)    | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round5](yale-nlp/comal-qwen2-1.5b-iter-ipo-round5) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round5](yale-nlp/comal-qwen2-1.5b-inpo-small-round5) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round5](yale-nlp/comal-qwen2-1.5b-inpo-large-round5) | [yale-nlp/comal-qwen2-1.5b-round5](yale-nlp/comal-qwen2-1.5b-round5) |
| 6 (36)    | [yale-nlp/comal-qwen2-1.5b-iter-ipo-round6](yale-nlp/comal-qwen2-1.5b-iter-ipo-round6) | [yale-nlp/comal-qwen2-1.5b-inpo-small-round6](yale-nlp/comal-qwen2-1.5b-inpo-small-round6) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round6](yale-nlp/comal-qwen2-1.5b-inpo-large-round6) | [yale-nlp/comal-qwen2-1.5b-round6](yale-nlp/comal-qwen2-1.5b-round6) |
| 7 (42)    | WIP | [yale-nlp/comal-qwen2-1.5b-inpo-small-round7](yale-nlp/comal-qwen2-1.5b-inpo-small-round7) | [yale-nlp/comal-qwen2-1.5b-inpo-large-round7](yale-nlp/comal-qwen2-1.5b-inpo-large-round7) | [yale-nlp/comal-qwen2-1.5b-round7](yale-nlp/comal-qwen2-1.5b-round7) |

## File Structure

### Files

- [`comal.sh`](comal.sh): Script for running COMAL.
- [`data_processing.py`](data_processing.py): Contains the code for post-processing the preference model annotations into training data for COMAL.
- [`data_utils.py`](data_utils.py): Utility functions for training data loading.
- [`eval.py`](eval.py): Evaluation script for COMAL.
- [`get_logprobs.py`](get_logprobs.py): Script for extracting log probabilities from an LLM/policy.
- [`losses.py`](losses.py): Loss functions for training COMAL.
- [`dpo.py`](dpo.py): DPO training.
- [`ipo.py`](ipo.py): IPO training.
- [`mle.py`](mle.py): MLE training.
- [`inpo.py`](inpo.py): INPO training.
- [`sampling.py`](sampling.py): Sampling candidate outputs from an LLM.
- [`scoring.py`](scoring.py): Scoring output pairs using a preference model.
- [`utils.py`](utils.py): Utility functions.
- [`vllm_model.py`](vllm_model.py): VLLM model definition.
- [`fsdp_config.yaml`](fsdp_config.yaml): Configuration file for FSDP.

### Directories
- [`data/prompts`](data/prompts): Contains the prompts used for training and evaluation. The prompts are from the [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset. Please cite their [work](https://arxiv.org/abs/2310.01377) if you use these prompts.
- [`exps`](exps): Contains the results of the experiments. A new directory is created for each experiment, with the name specified in `comal.sh`.
- [`prompts`](prompts): Contains the prompt used for the preference model.
