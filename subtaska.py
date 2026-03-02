import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A/train.parquet')
val = pd.read_parquet('/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A/validation.parquet')
test = pd.read_parquet('/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A/test.parquet')

print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")
print(f"Test shape: {test.shape}")

# Feature Engineering
def extract_code_features(code):
    """Extract statistical and stylistic features from code"""
    features = {}
   
    # Basic statistics
    features['length'] = len(code)
    features['num_lines'] = code.count('\n') + 1
    features['avg_line_length'] = len(code) / (code.count('\n') + 1)
   
    # Whitespace patterns
    features['spaces'] = code.count(' ')
    features['tabs'] = code.count('\t')
    features['space_ratio'] = features['spaces'] / (len(code) + 1)
    features['newlines'] = code.count('\n')
   
    # Comments
    features['single_comments'] = len(re.findall(r'//', code)) + len(re.findall(r'#', code))
    features['multi_comments'] = len(re.findall(r'/\*.*?\*/', code, re.DOTALL))
   
    # Code structure
    features['brackets'] = code.count('{') + code.count('}')
    features['parentheses'] = code.count('(') + code.count(')')
    features['semicolons'] = code.count(';')
    features['colons'] = code.count(':')
   
    # Common keywords (language-agnostic)
    keywords = ['if', 'else', 'for', 'while', 'return', 'function', 'def', 'class',
                'import', 'include', 'public', 'private', 'void', 'int', 'string']
    for kw in keywords:
        features[f'kw_{kw}'] = len(re.findall(r'\b' + kw + r'\b', code.lower()))
   
    # Variable naming patterns
    camel_case = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', code))
    snake_case = len(re.findall(r'\b[a-z]+_[a-z_]+\b', code))
    features['camel_case'] = camel_case
    features['snake_case'] = snake_case
   
    # Identifier length statistics
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    if identifiers:
        id_lengths = [len(x) for x in identifiers]
        features['avg_identifier_len'] = np.mean(id_lengths)
        features['max_identifier_len'] = np.max(id_lengths)
        features['unique_identifiers'] = len(set(identifiers))
    else:
        features['avg_identifier_len'] = 0
        features['max_identifier_len'] = 0
        features['unique_identifiers'] = 0
   
    # Indentation consistency
    lines = code.split('\n')
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if indents:
        features['avg_indent'] = np.mean(indents)
        features['std_indent'] = np.std(indents)
    else:
        features['avg_indent'] = 0
        features['std_indent'] = 0
   
    # Character n-grams diversity
    char_bigrams = [code[i:i+2] for i in range(len(code)-1)]
    features['bigram_diversity'] = len(set(char_bigrams)) / (len(char_bigrams) + 1)
   
    # Punctuation density
    punctuation = '!@#$%^&*()_+-=[]{}|;:,.<>?/'
    features['punct_density'] = sum(code.count(p) for p in punctuation) / (len(code) + 1)
   
    return features

print("Extracting features from training data...")
train_features = pd.DataFrame([extract_code_features(code) for code in train['code']])
val_features = pd.DataFrame([extract_code_features(code) for code in val['code']])
test_features = pd.DataFrame([extract_code_features(code) for code in test['code']])

# TF-IDF features with character n-grams (better for OOD)
print("Creating TF-IDF features...")
tfidf_word = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    analyzer='word',
    token_pattern=r'\b\w+\b',
    min_df=2,
    max_df=0.95
)

tfidf_char = TfidfVectorizer(
    max_features=2000,
    ngram_range=(3, 5),
    analyzer='char',
    min_df=2,
    max_df=0.95
)

# Fit and transform
X_train_word = tfidf_word.fit_transform(train['code'])
X_val_word = tfidf_word.transform(val['code'])
X_test_word = tfidf_word.transform(test['code'])

X_train_char = tfidf_char.fit_transform(train['code'])
X_val_char = tfidf_char.transform(val['code'])
X_test_char = tfidf_char.transform(test['code'])

# Combine all features
from scipy.sparse import hstack

X_train = hstack([X_train_word, X_train_char, train_features.values])
X_val = hstack([X_val_word, X_val_char, val_features.values])
X_test = hstack([X_test_word, X_test_char, test_features.values])

y_train = train['label'].values
y_val = val['label'].values

print(f"Final feature shape: {X_train.shape}")

# Train ensemble models
print("\nTraining models...")

# Model 1: Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
rf_pred_val = rf.predict_proba(X_val)[:, 1]
rf_pred_test = rf.predict_proba(X_test)[:, 1]

# Model 2: Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred_val = gb.predict_proba(X_val)[:, 1]
gb_pred_test = gb.predict_proba(X_test)[:, 1]

# Model 3: Logistic Regression (good for generalization)
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr.fit(X_train, y_train)
lr_pred_val = lr.predict_proba(X_val)[:, 1]
lr_pred_test = lr.predict_proba(X_test)[:, 1]

# Ensemble predictions with optimized weights
weights = [0.35, 0.35, 0.30]  # RF, GB, LR
val_pred_proba = (weights[0] * rf_pred_val +
                  weights[1] * gb_pred_val +
                  weights[2] * lr_pred_val)
test_pred_proba = (weights[0] * rf_pred_test +
                   weights[1] * gb_pred_test +
                   weights[2] * lr_pred_test)

# Optimal threshold tuning on validation set
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    val_pred = (val_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_val, val_pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nBest threshold: {best_threshold:.3f}")
print(f"Validation Macro F1: {best_f1:.4f}")

# Individual model scores
rf_val_pred = (rf_pred_val >= best_threshold).astype(int)
gb_val_pred = (gb_pred_val >= best_threshold).astype(int)
lr_val_pred = (lr_pred_val >= best_threshold).astype(int)
ensemble_val_pred = (val_pred_proba >= best_threshold).astype(int)

print(f"\nRandom Forest F1: {f1_score(y_val, rf_val_pred, average='macro'):.4f}")
print(f"Gradient Boosting F1: {f1_score(y_val, gb_val_pred, average='macro'):.4f}")
print(f"Logistic Regression F1: {f1_score(y_val, lr_val_pred, average='macro'):.4f}")
print(f"Ensemble F1: {f1_score(y_val, ensemble_val_pred, average='macro'):.4f}")

# Generate predictions for test set
test_pred = (test_pred_proba >= best_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'label': test_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")
print(f"Predictions distribution: {Counter(test_pred)}")
print(submission.head())
