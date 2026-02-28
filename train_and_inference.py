# ============================================================
# SemEval 2026 - Task 13 - Subtask C
# Detection of AI-generated, Hybrid and Adversarial Code
# Trained on Google Colab T4 GPU
# ============================================================

# !pip install -q transformers accelerate

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from google.colab import drive
drive.mount('/content/drive')


# =========================
# 1. Load Dataset
# =========================

BASE_DIR = "/content/drive/MyDrive/Task_C/"

train_path = os.path.join(BASE_DIR, "train.parquet")
val_path   = os.path.join(BASE_DIR, "validation.parquet")
test_path  = os.path.join(BASE_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
val_df   = pd.read_parquet(val_path)
test_df  = pd.read_parquet(test_path)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nTrain label distribution:")
print(train_df["label"].value_counts().sort_index())


# =========================
# 2. Downsample Human Class
# =========================

df_human = train_df[train_df["label"] == 0]
df_other = train_df[train_df["label"] != 0]

KEEP_HUMAN = 40000

rng = np.random.default_rng(42)
idx = rng.choice(df_human.index.values, size=KEEP_HUMAN, replace=False)
df_human_sampled = df_human.loc[idx]

train_balanced_df = pd.concat(
    [df_human_sampled, df_other],
    ignore_index=True
).sample(frac=1.0, random_state=42).reset_index(drop=True)

print("\nBalanced label distribution:")
print(train_balanced_df["label"].value_counts())


# =========================
# 3. Text Construction
# =========================

MAX_CHARS = 800

def make_text_for_transformer(df):
    if "language" in df.columns:
        lang = df["language"].fillna("")
    else:
        lang = pd.Series([""] * len(df), index=df.index)

    code = df["code"].fillna("").str.slice(0, MAX_CHARS)
    return (lang + " " + code).astype(str).tolist()


train_texts = make_text_for_transformer(train_balanced_df)
train_labels = train_balanced_df["label"].to_numpy().astype(int)

val_texts = make_text_for_transformer(val_df)
val_labels = val_df["label"].to_numpy().astype(int)

test_texts = make_text_for_transformer(test_df)


# =========================
# 4. Model Setup
# =========================

MODEL_NAME = "microsoft/codebert-base"
MAX_TOKENS = 256
BATCH_SIZE = 12
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


data_collator = DataCollatorWithPadding(tokenizer)

train_dataset = CodeDataset(train_texts, train_labels, tokenizer, MAX_TOKENS)
val_dataset   = CodeDataset(val_texts, val_labels, tokenizer, MAX_TOKENS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)


# =========================
# 5. Training Setup
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

scaler = torch.cuda.amp.GradScaler()


# =========================
# 6. Evaluation
# =========================

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return f1_score(all_labels, all_preds, average="macro")


# =========================
# 7. Training Loop
# =========================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: val_macro_f1 = {val_f1:.4f}")


# =========================
# 8. Inference
# =========================

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in enc.items()}


test_dataset = TestDataset(test_texts, tokenizer, MAX_TOKENS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.append(preds.cpu().numpy())

all_preds = np.concatenate(all_preds)


# =========================
# 9. Submission
# =========================

submission = pd.DataFrame({
    "id": np.arange(len(test_df)),
    "label": all_preds
})

submission.to_csv("submission_subtaskC.csv", index=False)

from google.colab import files
files.download("submission_subtaskC.csv")
