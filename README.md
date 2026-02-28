# SemEval-2026 Task 13 – Subtask C  
## Detection of AI-Generated, Hybrid, and Adversarial Code

This repository contains our implementation for **SemEval-2026 Task 13, Subtask C**, which focuses on detecting nuanced authorship patterns in source code.

The task extends beyond simple AI detection and introduces hybrid and adversarial code scenarios, reflecting real-world AI-assisted programming practices.

---

## 1. Problem Definition

Given a source code snippet, the objective is to classify it into one of four categories:

| Label | Class |
|-------|--------|
| 0 | Fully human-written |
| 1 | Fully AI-generated |
| 2 | Hybrid (human + LLM edited/completed) |
| 3 | Adversarial (LLM-generated to mimic human style) |

This is a **4-class multi-class classification problem**, evaluated using **Macro F1 score**.

---

## 2. Motivation

As AI-assisted coding becomes widespread:

- Students draft assignments and refine them using LLMs
- Job candidates submit AI-assisted solutions
- Developers rely on partial completions and stylistic rewrites
- Advanced LLMs can be prompted or trained to evade detection

Subtask C introduces **hybrid and adversarial code**, making it significantly more challenging than binary AI detection.

This task aims to build detection systems that are robust to:
- Style mimicry
- Partial human edits
- Adversarial prompt engineering

---

## 3. Dataset Description

### Files

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `test_sample.parquet`
- `sample_submission.csv`

### Columns

| Column | Description |
|--------|-------------|
| code | Program source code |
| generator | Model name or "human" |
| language | Programming language |
| label | 0–3 class label |

### Dataset is available on competition website (link in about)

### Class Distribution (Before Balancing)

The dataset is naturally imbalanced, with human samples dominating.

To mitigate bias, we applied controlled downsampling (see Section 5).

---

## 4. Methodology

### 4.1 Text Representation

Each sample is constructed as:

```
<language> + " " + first 800 characters of code
```

This preserves:

- Language-specific syntax
- Early structural patterns
- Signature generation artifacts

Truncation prevents memory explosion while retaining stylistic cues.

---

### 4.2 Model Architecture

We fine-tuned:

```
microsoft/codebert-base
```

CodeBERT is a Transformer-based model pre-trained on:

- Natural language
- Programming languages
- Bimodal NL-PL objectives

Its architecture is based on RoBERTa.

---

### 4.3 Training Configuration

| Parameter | Value |
|------------|--------|
| Max Tokens | 256 |
| Max Characters | 800 |
| Batch Size | 12 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Scheduler | Linear warmup (10%) |
| Precision | Mixed Precision (AMP) |
| GPU | NVIDIA T4 (Google Colab) |

Dynamic padding was used via `DataCollatorWithPadding` to improve memory efficiency.

---

### 4.4 Class Imbalance Handling

Instead of naive training, we applied:

1. Downsampling of human class:
   - Reduced to 40,000 samples

2. Weighted loss function:
   - Class weights computed using `compute_class_weight`
   - Applied via weighted CrossEntropyLoss

This ensured balanced gradient contributions across classes.

---

## 5. Training Strategy

We trained using mixed precision (AMP) to:

- Reduce memory footprint
- Increase training throughput
- Stabilize training on T4 GPU

Macro F1 was used as evaluation metric due to imbalance.

---

## 6. Results

### Validation Performance

| Epoch | Validation Macro F1 |
|--------|--------------------|
| 1 | 0.7702 |
| 2 | 0.7626 |
| 3 | **0.7774** |

### Official Leaderboard Score

| Setting | Macro F1 |
|----------|----------|
| Public Leaderboard | **0.53134** |

Metric: **Macro F1**

---

### Performance Analysis

The gap between validation (0.7774) and leaderboard (0.53134) suggests:

- Distribution shift between train/validation and hidden test set
- Increased difficulty in adversarial samples
- Potential unseen generation strategies
- Stronger hybrid/human overlap in final test data

Despite the shift, the model maintains stable multi-class discrimination under adversarial conditions.

---

## 7. Observations

- Hybrid samples are harder than fully AI-generated samples.
- Adversarial examples significantly overlap stylistically with human code.
- Language token inclusion improves performance.
- Class-weighted loss stabilizes minority class learning.

---

## 8. Limitations

- Context truncated to 800 characters
- No hierarchical modeling
- No adversarial training defense
- No ensemble methods

---

## 9. Future Work

Potential improvements:

- Hierarchical classification (Human vs AI → Subtype)
- Contrastive learning for stylistic separation
- Ensemble of code-specific models
- Prompt-invariant detection features
- Structural AST-based augmentation

---

## 10. Reproducibility

### Install dependencies

```
pip install -r requirements.txt
```

### Train and generate submission

```
python train_and_inference.py
```

Ensure dataset files are placed in:

```
/content/drive/MyDrive/Task_C/
```

---

## 11. Repository Structure

```
SemEval-2026-Task13-Subtask-C/
│
├── README.md
├── train_and_inference.py
├── requirements.txt
```
