##### Imports #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from tqdm import tqdm
import umap
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

import os
#Check for all necessary files
os.listdir()


#Reading and Preparing Data
df_content = pd.read_table("reviewContent", header=None, names=["user_id", "prod_id", "date", "Review"])
df_content.head()

df_meta = pd.read_table("metadata", header=None, names=["user_id", "prod_id", "rating", "label", "date"])
df_meta.head()

df_labels = pd.read_table("reviewGraph", header=None, names=["ID", "?", "Rating"])
df_labels.head()


##### Merge dataframes, Check for missing values, Splitting the data #####
df_all = df_content.merge(
    df_meta,
    on=["user_id", "prod_id", "date"],
    how="left"
)
df_all.head()

print(df_all.isna().sum())
df_95, df_5 = train_test_split(
    df_all,
    test_size=0.01, # CAN ADJUST THIS ID IT DOES NOT RUN
    stratify=df_all["label"],
)

df_5["label_name"] = df_5["label"].map({-1: "Spam", 1: "Non‑Spam"})
pct = df_5["label_name"].value_counts(normalize=True) * 100

#Class distribution graph
sns.barplot(x=pct.index, y=pct.values, color = "#C8A2C8")
plt.ylabel("Percentage (%)")
plt.xlabel("Class")
plt.title("Class Distribution (%)")


##### DeBERTa Embedding #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

def get_deberta_embedding(text, batch_size = 500):
    all_embeddings = []

    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_emb.cpu().numpy())

    return np.vstack(all_embeddings)


X_deb = get_deberta_embedding(df_5["Review"].tolist())

df_5["deberta_emb"] = list(X_deb)

df_5.to_csv("DebertEmbed.csv", index=False)

reducer = umap.UMAP()

embedding = reducer.fit_transform(np.vstack(df_5["deberta_emb"].values))
embedding.shape

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = df_5["label"])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the DeBERTa model', fontsize=24);
plt.show()


##### CNN Embedding #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_channels=100, kernel_sizes=[3,4,5]):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_channels, k)
            for k in kernel_sizes
        ])

        self.output_dim = num_channels * len(kernel_sizes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                 # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))            # (batch, channels, L')
            p = torch.max(c, dim=2).values     # (batch, channels)
            conv_outs.append(p)

        return torch.cat(conv_outs, dim=1)     # (batch, output_dim)


model_name = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

cnn = SimpleTextCNN(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    kernel_sizes=[3,4,5],
    num_channels=100
).to(device)

batch_size = 64
all_embeddings = []

for i in tqdm(range(0, len(df_5), batch_size)):
    batch_text = df_5["Review"].iloc[i:i+batch_size].tolist()

    inputs = tokenizer(
        batch_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    ).to(device)
    input_ids = inputs["input_ids"].to(device)


    with torch.no_grad():
        batch_emb = cnn(inputs["input_ids"])   # (batch, 300)

    all_embeddings.append(batch_emb.cpu().numpy())


cnn_emb = np.vstack(all_embeddings)
df_5["cnn_emb"] = list(cnn_emb)

reducer2 = umap.UMAP()

embedding = reducer2.fit_transform(np.vstack(df_5["cnn_emb"].values))
embedding.shape

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = df_5["label"])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the CNN model', fontsize=24);
plt.show()


##### Feature Extraction #####
df_5["date"] = pd.to_datetime(df_5["date"])
df_5 = df_5.sort_values(["user_id", "date"])
df_5["time_since_prev"] = (df_5.groupby("user_id")["date"].diff().dt.total_seconds())
df_5["time_since_prev"] = df_5["time_since_prev"].fillna(0)

df_5 = df_5.sort_values(["prod_id", "date"])
df_5["time_since_prev_prod"] = (df_5.groupby("prod_id")["date"].diff().dt.total_seconds())
df_5["time_since_prev_prod"] = df_5["time_since_prev_prod"].fillna(0)

for col in ["time_since_prev", "time_since_prev_prod"]:
    df_5[col] = (df_5[col] - df_5[col].min()) / (df_5[col].max() - df_5[col].min())

X_deb = np.vstack(df_5["deberta_emb"].values)
X_cnn = np.vstack(df_5["cnn_emb"].values)
review_len = df_5["Review"].str.len().to_numpy().reshape(-1, 1)
word_count = df_5["Review"].str.split().str.len().to_numpy().reshape(-1, 1)
timestamp_features = df_5[["time_since_prev", "time_since_prev_prod"]].to_numpy()
prod_means = df_5.groupby('prod_id')['rating'].transform('mean')
rating_dev = np.abs(df_5['rating'] - prod_means).to_numpy().reshape(-1, 1)
prod_target_count = df_5.groupby('prod_id')['Review'].transform('count').to_numpy().reshape(-1, 1)

df_5["review_len"] = df_5["Review"].str.len().to_numpy().reshape(-1, 1)
df_5["word_count"] = df_5["Review"].str.split().str.len().to_numpy().reshape(-1, 1)
df_5["prod_means"] = df_5.groupby('prod_id')['rating'].transform('mean')
df_5["rating_dev"] = np.abs(df_5['rating'] - prod_means).to_numpy().reshape(-1, 1)
df_5["prod_target_count"] = df_5.groupby('prod_id')['Review'].transform('count').to_numpy().reshape(-1, 1)



df_5["user_review_count"] = df_5.groupby("user_id")["Review"].transform("count")
df_5["user_rating_var"] = df_5.groupby("user_id")["rating"].transform("var").fillna(0)
df_5["user_avg_rating"] = df_5.groupby("user_id")["rating"].transform("mean")
df_5["user_time_mean"] = df_5.groupby("user_id")["time_since_prev"].transform("mean")
df_5["user_time_std"] = df_5.groupby("user_id")["time_since_prev"].transform("std").fillna(0)
df_5["user_prod_review_count"] = (
    df_5.groupby(["user_id", "prod_id"])["Review"].transform("count")
)

def entropy(x):
    x = x.astype(int)
    counts = np.bincount(x, minlength=6)[1:]  # ratings 1–5
    probs = counts / counts.sum()
    return -(probs[probs > 0] * np.log(probs[probs > 0])).sum()

df_5["user_rating_entropy"] = (
    df_5.groupby("user_id")["rating"]
        .transform(lambda x: entropy(x.to_numpy()))
)

df_5["prod_rating_var"] = (
    df_5.groupby("prod_id")["rating"]
        .transform("var")
        .fillna(0)
)


X_beh_raw = np.hstack([review_len, word_count, timestamp_features,
                       prod_target_count, rating_dev,
                          df_5[["user_review_count", "user_avg_rating",
                                "user_rating_var","user_time_mean",
                                "user_time_std", "user_rating_entropy",
                                "prod_rating_var", "user_prod_review_count"]].to_numpy()
                      ])
X_beh = StandardScaler().fit_transform(X_beh_raw)

deb_scaler = StandardScaler()
X_deb_std = deb_scaler.fit_transform(X_deb)
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_deb_pca = pca.fit_transform(X_deb_std)

X_combined = np.hstack([X_deb_std * 0.3, X_beh * 1])


fig, axes = plt.subplots(5, 3, figsize=(18, 18))
axes = axes.flatten()
features = [
    "review_len",
    "word_count",
    "prod_target_count",
    "rating_dev",
    "user_review_count",
    "user_avg_rating",
    "user_rating_var",
    "user_time_mean",
    "user_time_std",
    "user_rating_entropy",
    "prod_rating_var",
    "user_prod_review_count",
    "time_since_prev",
    "time_since_prev_prod"
]

for ax, feature in zip(axes, features):
    sns.boxplot(
        data=df_5,
        x="label_name",
        y=feature,
        ax=ax
    )
    ax.set_title(f"{feature} by Class", fontsize=12)
    ax.set_xlabel("Class")
    ax.set_ylabel(feature)


plt.tight_layout()
plt.show()


##### Combining Embeddings for Clustering #####
W1 = kneighbors_graph(X_deb, n_neighbors=50, mode='connectivity')
W2 = kneighbors_graph(X_beh, n_neighbors=50, mode='connectivity')
W3 = kneighbors_graph(X_cnn, n_neighbors=50, mode='connectivity')

deg1 = W1.sum() / W1.shape[0]
deg2 = W2.sum() / W2.shape[0]
deg3 = W3.sum() / W3.shape[0]

w1_wgt = 1 / deg1
w2_wgt = 1 / deg2
w3_wgt = 1 / deg3
sum = w1_wgt + w2_wgt + w3_wgt
w1_wgt, w2_wgt, w3_wgt = w1_wgt/sum, w2_wgt/sum, w3_wgt/sum

W_fused = w1_wgt*W1 + w2_wgt*W2 + w3_wgt*W3


###### Multiview Specteral Clustering ######
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(
    n_clusters=2,
    affinity='precomputed',
    assign_labels='cluster_qr'
)

df_5["labels_msc"]= sc.fit_predict(W_fused)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(df_5["label"], df_5["labels_msc"])
nmi = normalized_mutual_info_score(df_5["label"], df_5["labels_msc"])

print("ARI:", ari)
print("NMI:", nmi)

reducer = umap.UMAP()
embedding = reducer.fit_transform(X_beh)

plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df_5["label"], cmap='coolwarm', alpha=0.7)
plt.title('UMAP: Pure Behavioral View (No Stylometry)', fontsize=15)
plt.colorbar(label='Ground Truth Label')
plt.show()

def objective(trial):
    # 1. Suggest weights for the three views (must sum to 1.0)
    w1 = trial.suggest_float("w1", 0.1, 0.8)
    w2 = trial.suggest_float("w2", 0.1, 0.8)
    w3 = trial.suggest_float("w3", 0.1, 0.8)

    # Normalize weights so they sum to 1
    total = w1 + w2 + w3
    w1, w2, w3 = w1/total, w2/total, w3/total

    # 2. Fuse the graphs with these trial weights
    W_fused_trial = (w1 * W1) + (w2 * W2) + (w3 * W3)

    # 3. Run Spectral Clustering
    # Note: We use a fixed random_state to ensure consistency
    sc_trial = SpectralClustering(
        n_clusters=2,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )

    try:
        labels = sc_trial.fit_predict(W_fused_trial)
        # 4. Objective: Maximize ARI (how well we match ground truth)
        score = adjusted_rand_score(df_5["label"], labels)
    except:
        score = 0 # Handle cases where clustering fails to converge

    return score

# Create the study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10) # Increase n_trials for better results

print("\n--- OPTUNA MBO RESULTS ---")
print(f"Best ARI Score: {study.best_value:.4f}")
print(f"Best Weights: {study.best_params}")

# Apply the best weights to your final labels
best_w = study.best_params
w_sum = sum(best_w.values())
W_optimized = (best_w['w1']/w_sum * W1) + (best_w['w2']/w_sum * W2) + (best_w['w3']/w_sum * W3)

df_5['mbo_labels'] = SpectralClustering(
    n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=42
).fit_predict(W_optimized)

pd.crosstab(df_5['label_name'], df_5['mbo_labels'])


###### Classification ######
df_export = pd.DataFrame(X_combined)
df_export.to_csv("X_combined.csv", index=False)

X = X_combined
y = df_5["label"].values

df_5["label_fixed"] = df_5["label"].replace({-1: 0, 1: 1})

X_train, X_val, y_train, y_val = train_test_split(
    X_combined,
    df_5["label_fixed"],
    test_size=0.2,
    stratify=df_5["label_fixed"]
)

######## XGBoost ########
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
spw = neg / pos

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": spw,
        "eval_metric": "auc",
        "tree_method": "hist"
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)

    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)

best = study.best_params

xgb = XGBClassifier(
    n_estimators=best["n_estimators"],
    max_depth=best["max_depth"],
    learning_rate=best["learning_rate"],
    subsample=best["subsample"],
    colsample_bytree=best["colsample_bytree"],
    min_child_weight = best["min_child_weight"],
    gamma = best["gamma"],
    scale_pos_weight=spw,
    eval_metric="auc",
    tree_method="hist"
)

xgb.fit(X_train, y_train)

pred_proba = xgb.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc_xg = [balanced_accuracy_score(y_val, pred_proba >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc_xg)]

pred = (pred_proba >= best_threshold).astype(int)

xgb_auc = roc_auc_score(y_val, pred_proba)
xgb_f1 = f1_score(y_val, pred)
xgb_acc = accuracy_score(y_val, pred)
xgb_balacc = balanced_accuracy_score(y_val, pred)

xgb_test_results = {
    "Accuracy": xgb_acc,
    "Balanced Accuracy": xgb_balacc,
    "F1 Score": xgb_f1,
    "ROC AUC": xgb_auc
}

print("XGBOOST")
print("Best threshold (F1):", best_threshold)
print("AUC:", xgb_auc)
print("F1:", xgb_f1)
print("Accuracy:", xgb_acc)

from sklearn.metrics import roc_curve, auc

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Logistic Regression ########
def objective(trial):

    C = trial.suggest_float("C", 1e-4, 10.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else trial.suggest_categorical("solver", ["liblinear", "saga"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    LR = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=2000
    )

    LR.fit(X_train, y_train)

    preds = LR.predict_proba(X_val)[:, 1]
    preds_raw = LR.predict(X_val)

    score = roc_auc_score(y_val, preds)

    return score

study_LR = optuna.create_study(direction="maximize")
study_LR.optimize(objective, n_trials=10)

print("Best Score:", study_LR.best_value)
print("Best Params:", study_LR.best_params)


best_LR = study_LR.best_params

final_model = LogisticRegression(
    C=best_LR["C"],
    penalty=best_LR["penalty"],
    solver="liblinear" if best_LR["penalty"] == "l1" else best_LR["solver"],
    class_weight=best_LR["class_weight"],
    max_iter=2000,
    random_state=42
)

final_model.fit(X_train, y_train)

pred_proba_LR = final_model.predict_proba(X_val)[:, 1]

thresholds_LR = np.linspace(0.01, 0.99, 200)
balacc_LR = [balanced_accuracy_score(y_val, pred_proba_LR >= t) for t in thresholds]

best_threshold_LR = thresholds[np.argmax(balacc_LR)]

preds_raw_LR = (pred_proba_LR >= best_threshold_LR).astype(int)

lr_roc = roc_auc_score(y_val, pred_proba_LR)
lr_f1 = f1_score(y_val, preds_raw_LR)
lr_accuracy = accuracy_score(y_val, preds_raw_LR)
lr_balanced_acc = balanced_accuracy_score(y_val, preds_raw_LR)

lr_test_results = {
    "Accuracy": lr_accuracy,
    "Balanced Accuracy": lr_balanced_acc,
    "F1 Score": lr_f1,
    "ROC AUC": lr_roc
}

print("Logistic Regression")
print("Best Threshold:", best_threshold)
print("ROC/AUC Score:", lr_roc)
print("F1 Score:", lr_f1)
print("Accuracy:", lr_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba_LR)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## SVM ########
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

def objective(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)
    if kernel == "rbf":
        gamma = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
        degree = trial.suggest_int("degree", 1, 10)
        coef0 = 0.0

    elif kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
        gamma = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
        coef0 = trial.suggest_float("coef0", 0.0, 1.0)

    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])


    # Build model
    sv = SVC(
        kernel = kernel,
        degree = degree,
        gamma = gamma,
        C=C,
        class_weight=class_weight,
        probability = True,
        max_iter=20000,
    )

    sv.fit(X_train_pca, y_train)

    sv_preds = sv.predict_proba(X_val_pca)[:, 1]

    score = roc_auc_score(y_val, sv_preds)

    return score

study_sv = optuna.create_study(direction="maximize")
study_sv.optimize(objective, n_trials=10)

print("Best Score:", study_sv.best_value)
print("Best Params:", study_sv.best_params)

best_sv = study_sv.best_params

SVC_model = SVC(
    kernel = best_sv["kernel"],
    degree = best_sv["degree"],
    gamma = best_sv["gamma"],
    C=best_sv["C"],
    class_weight=best_sv["class_weight"],
    probability = True,
    max_iter=2000,
)

SVC_model.fit(X_train_pca, y_train)
pred_proba_sv = SVC_model.predict_proba(X_val_pca)[:, 1]
 
thresholds = np.linspace(0.01, 0.99, 200)
balacc_sv = [balanced_accuracy_score(y_val, pred_proba_sv >= t) for t in thresholds]

best_threshold_sv = thresholds[np.argmax(balacc_sv)]

sv_preds_raw = (pred_proba_sv >= best_threshold_sv).astype(int)

svm_roc = roc_auc_score(y_val, pred_proba_sv)
svm_f1 = f1_score(y_val, sv_preds_raw)
svm_accuracy = accuracy_score(y_val, sv_preds_raw)
svm_balanced_acc = balanced_accuracy_score(y_val, sv_preds_raw)

svm_test_results = {
    "Accuracy": svm_accuracy,
    "Balanced Accuracy": svm_balanced_acc,
    "F1 Score": svm_f1,
    "ROC AUC": svm_roc
}

print("SVM")
print("Best Threshold:", best_threshold_sv)
print("ROC/AUC Score:", svm_roc)
print("F1 Score:", svm_f1)
print("Accuracy:", svm_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba_sv)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Neural Network ########
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = tuple(trial.suggest_int(f"n_units_layer_{i}", 32, 256) for i in range(n_layers))
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
    alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    learning_rate= trial.suggest_categorical("learning_rate", ["constant", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

    # Build model
    model = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,
        activation = activation,
        solver = solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=50000
    )

    model.fit(X_train_pca, y_train)

    preds = model.predict_proba(X_val_pca)[:, 1]

    score = roc_auc_score(y_val, preds)


    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Score:", study.best_value)
print("Best Params:", study.best_params)

best_nn = study.best_params

nn_model = MLPClassifier(
    hidden_layer_sizes = tuple(best_nn[f"n_units_layer_{i}"] for i in range(best_nn["n_layers"])),
    activation = best_nn["activation"],
    solver = best_nn["solver"],
    alpha=best_nn["alpha"],
    learning_rate=best_nn["learning_rate"],
    learning_rate_init=best_nn["learning_rate_init"],
    max_iter=50000
)

nn_model.fit(X_train_pca, y_train)
pred_proba_nn = nn_model.predict_proba(X_val_pca)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc_nn = [balanced_accuracy_score(y_val, pred_proba_nn >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc_nn)]

preds_raw_nn = (pred_proba_nn >= best_threshold).astype(int)

nn_roc = roc_auc_score(y_val, pred_proba_nn)
nn_f1 = f1_score(y_val, preds_raw_nn)
nn_accuracy = accuracy_score(y_val, preds_raw_nn)
nn_balanced_acc = balanced_accuracy_score(y_val, preds_raw_nn)

nn_test_results = {
    "Accuracy": nn_accuracy,
    "Balanced Accuracy": nn_balanced_acc,
    "F1 Score": nn_f1,
    "ROC AUC": nn_roc
}

print("NN Classifier")
print("Best Threshold:", best_threshold)
print("ROC/AUC Score:", nn_roc)
print("F1 Score:", nn_f1)
print("Accuracy:", nn_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba_nn)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Neural Network")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Decision Tree ########
def objective(trial):
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)
    return auc

study_DT = optuna.create_study(direction="maximize")
study_DT.optimize(objective, n_trials=10)

print("Best AUC:", study_DT.best_value)
print("Best Params:", study_DT.best_params)

best_dt_params = study_DT.best_params

dt_model = DecisionTreeClassifier(
    criterion=best_dt_params["criterion"],
    max_depth=best_dt_params["max_depth"],
    min_samples_split=best_dt_params["min_samples_split"],
    min_samples_leaf=best_dt_params["min_samples_leaf"],
    max_features=best_dt_params["max_features"],
    class_weight=best_dt_params["class_weight"],
)

dt_model.fit(X_train, y_train)
pred_proba = dt_model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc = [balanced_accuracy_score(y_val, pred_proba >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc)]

preds_raw = (pred_proba >= best_threshold).astype(int)

dt_roc = roc_auc_score(y_val, pred_proba)
dt_f1 = f1_score(y_val, preds_raw)
dt_accuracy = accuracy_score(y_val, preds_raw)
dt_balanced_acc = balanced_accuracy_score(y_val, preds_raw)

dt_test_results = {
    "Accuracy": dt_accuracy,
    "Balanced Accuracy": dt_balanced_acc,
    "F1 Score": dt_f1,
    "ROC AUC": dt_roc
}

print("Decision Tree")
print("Best Threshold:", best_threshold)
print("ROC/AUC Score:", dt_roc)
print("F1 Score:", dt_f1)
print("Accuracy:", dt_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Random Forest ########
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)

best_rf_params = study.best_params

rf_model = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"],
    max_depth=best_rf_params["max_depth"],
    min_samples_split=best_rf_params["min_samples_split"],
    min_samples_leaf=best_rf_params["min_samples_leaf"],
    max_features=best_rf_params["max_features"],
    class_weight=best_rf_params["class_weight"],
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
pred_proba = rf_model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc = [balanced_accuracy_score(y_val, pred_proba >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc)]

preds_raw = (pred_proba >= best_threshold).astype(int)

rf_roc = roc_auc_score(y_val, pred_proba)
rf_f1 = f1_score(y_val, preds_raw)
rf_accuracy = accuracy_score(y_val, preds_raw)
rf_balanced_acc = balanced_accuracy_score(y_val, preds_raw)

rf_test_results = {
    "Accuracy": rf_accuracy,
    "Balanced Accuracy": rf_balanced_acc,
    "F1 Score": rf_f1,
    "ROC AUC": rf_roc
}

print("Random Forest")
print("Best Threshold:", best_threshold)
print("ROC/AUC Score:", rf_roc)
print("F1 Score:", rf_f1)
print("Accuracy:", rf_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Naive Bayes ########
from sklearn.naive_bayes import GaussianNB

def objective(trial):
    params = {
        "var_smoothing": trial.suggest_loguniform("var_smoothing", 1e-12, 1e-6)
    }

    model = GaussianNB(**params)
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)

best_nb_params = study.best_params

nb_model = GaussianNB(
    var_smoothing=best_nb_params["var_smoothing"]
)

nb_model.fit(X_train, y_train)

pred_proba = nb_model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc = [balanced_accuracy_score(y_val, pred_proba >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc)]

preds_raw = (pred_proba >= best_threshold).astype(int)

nb_roc = roc_auc_score(y_val, pred_proba)
nb_f1 = f1_score(y_val, preds_raw)
nb_accuracy = accuracy_score(y_val, preds_raw)
nb_balanced_acc = balanced_accuracy_score(y_val, preds_raw)

nb_test_results = {
    "Accuracy": nb_accuracy,
    "Balanced Accuracy": nb_balanced_acc,
    "F1 Score": nb_f1,
    "ROC AUC": nb_roc
}

print("Naive Bayes")
print("Best Threshold:", best_threshold)
print("ROC/AUC Score:", nb_roc)
print("F1 Score:", nb_f1)
print("Accuracy:", nb_accuracy)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Naive Bayes")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


######## Stacking ########
from sklearn.ensemble import StackingClassifier

base_models = [
    ('svm', SVC_model),
    ('dt', dt_model),
    ('rf', rf_model),
    ('xgb', xgb),
    ('nb', nb_model),
    ('nn', nn_model)
]

meta_model = LogisticRegression(max_iter=500)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method="predict_proba",
    passthrough=False,              
    n_jobs=-1
)

# Fit
stacking_model.fit(X_train, y_train)

pred_proba = stacking_model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
balacc = [balanced_accuracy_score(y_val, pred_proba >= t) for t in thresholds]

best_threshold = thresholds[np.argmax(balacc)]

preds_raw = (pred_proba >= best_threshold).astype(int)

stack_test_results = {
    "Accuracy": accuracy_score(y_val, preds_raw),
    "Balanced Accuracy": balanced_accuracy_score(y_val, preds_raw),
    "F1 Score": f1_score(y_val, preds_raw),
    "ROC AUC": roc_auc_score(y_val, pred_proba)
}
# Evaluate
print("Stacking Model:")
print("Best Threshold:", best_threshold)
print("ROC AUC:", roc_auc_score(y_val, pred_proba))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, preds_raw))
print("F1:", f1_score(y_val, preds_raw))
print("Accuracy:", accuracy_score(y_val, preds_raw))

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#BFA2DB", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Stacking")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()



##### Summary of Regression Statistics #####
all_metrics = pd.DataFrame({
    "SVM": svm_test_results,
    "Neural Network": nn_test_results,
    "Logistic Regression": lr_test_results,
    "XGBoost": xgb_test_results,
    "Decision Tree": dt_test_results,
    "Random Forest": rf_test_results,
    "Naive Bayes": nb_test_results,
    "Stacking": stack_test_results
}).T
all_metrics.sort_values("ROC AUC", ascending = False)

sorted_df = all_metrics.sort_values("ROC AUC", ascending=False)

metric_names = sorted_df.index
metric_values = sorted_df["ROC AUC"].values  # 1D array

plt.figure(figsize=(8, 6))
plt.barh(metric_names, metric_values, color="#D8B7DD")
plt.xlabel("ROC AUC")
plt.title("ROC AUC by Model (Sorted)")
plt.xlim(0, 1)
plt.gca().invert_yaxis()  # highest at top
plt.tight_layout()
plt.show()
