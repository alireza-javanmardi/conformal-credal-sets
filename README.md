# Conformalized Credal Set Predictors

This repository contains the code implementation for our paper titled "Conformalized Credal Set Predictors".

## Abstract

Credal sets represent collections of probability distributions that serve as potential candidates for imprecisely known ground-truth distributions. In the realm of machine learning, they have garnered attention as a compelling formalism for uncertainty representation, primarily for their capability to encompass both aleatoric and epistemic uncertainty within predictions. Nevertheless, devising methods for learning credal set predictors remains a formidable challenge. In our paper, we harness conformal prediction techniques to address this challenge. Specifically, we propose a novel approach for predicting credal sets within the classification task, leveraging training data annotated by probability distributions. Our method inherits the coverage guarantees inherent to conformal prediction, ensuring that our conformal credal sets are valid with high probability, without imposing constraints on the model or the underlying distribution. We illustrate the effectiveness of our approach in the context of natural language inference, a domain characterized by high ambiguity, often requiring multiple annotations per example.

## Setup

To set up the conda environment, follow these steps:

```bash
conda create --name ENV_NAME python=3.9
conda activate ENV_NAME
pip install -r requirements.txt
```


To execute the code, such as for the chaosNLI dataset, proceed as follows:

Run the script for chaosNLI training and calibration, starting with seed 2 and first-order:
```bash
python chaos_NLI_training_calibration.py 2 first
```
Subsequently, execute the script for chaosNLI testing with seed 2 and alpha value 0.2:
```bash
python chaos_NLI_test.py 2 0.2
```
Ensure you replace ENV_NAME with the name you prefer for your environment. Adjust the commands according to your specific requirements.
