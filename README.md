# COMAL

This repository is for our paper "COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences".

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
