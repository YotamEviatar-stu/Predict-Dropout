"""
Course-dropout prediction
=========================

Task: train and validate a model that predicts whether a B2B course enrollment will be
dropped (`Dropped_Course`), so at-risk clients can be contacted before they cancel.

    Step 1  Clean the raw table (messy text, fake numbers, leaky columns)
    Step 2  Turn it into model features
    Step 3  Train and tune three models with 5-fold cross-validation
    Step 4  Evaluate them once on a held-out 20% and explain the best one
    Step 5  Refit the best model on all labelled rows and score new enrollments

Run:  python pipeline.py
Reads data/Train_Data.csv (and data/Test_Data_No_Target.csv if present).
Writes tables to results/ and the README figures to assets/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR, RESULTS, FIGURES = Path("data"), Path("results"), Path("assets")
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
TARGET, SEED = "Dropped_Course", 42

# One colour per model, used in every figure.
COLORS = {"Logistic Regression": "#eb6834", "Random Forest": "#1baf7a", "XGBoost": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight"})


# =====================================================================================
# Step 1 - Cleaning
# =====================================================================================
# What happens here:
#   - Hand-typed text is normalised: upper-case, symbols removed. One client type had
#     141 spellings ('SaaS & Software Houses', 'saas & software houses', 'S!aaS & ...').
#   - Impossible numbers that stand for "missing" become NaN:
#       Students_Count = 9999, Practical_Hours in {-5, -1, 5000, 10000}, Daily_Tuition_Cost = 5400.
#   - Columns that are dropped:
#       Physical_Course_Kits, Assigned_Lab_Config -> only filled in once a client shows up,
#                                                    so they leak the answer
#       Welcome_Gift_Type, Lanyard_Color          -> no relation to dropping
#       Client_ID                                 -> unique per row
#   - 2 rows with Client_Category = UNKNOWN are removed.

def clean_text(series, keep=r"[^A-Z0-9\s]"):
    cleaned = (series.astype(str).str.upper()
               .str.replace(keep, "", regex=True)
               .str.replace(r"\s+", " ", regex=True).str.strip())
    return cleaned.where((cleaned != "") & series.notna(), "MISSING")


def clean(df):
    df = df.copy()
    df["country"] = clean_text(df["Origin_Country"], keep=r"[^A-Z]").replace("CN", "CHN")
    df["client_category"] = clean_text(df["Client_Category"], keep=r"[^A-Z\s]")
    for col in ["Payment_Terms", "Enrollment_Type", "Submission_Source", "Catering_Package"]:
        df[col.lower()] = clean_text(df[col])
    df["payment_terms"] = df["payment_terms"].where(
        df["payment_terms"].isin(["PAY UPON START", "PREPAID NONREFUNDABLE"]), "OTHER")

    df["Students_Count"] = df["Students_Count"].where(df["Students_Count"] != 9999)
    df["Practical_Hours"] = df["Practical_Hours"].where(~df["Practical_Hours"].isin([-5, -1, 5000, 10000]))
    df["Daily_Tuition_Cost"] = df["Daily_Tuition_Cost"].where(df["Daily_Tuition_Cost"] != 5400)
    df["Course_Start_Date"] = pd.to_datetime(df["Course_Start_Date"])
    return df


raw = pd.read_csv(DATA_DIR / "Train_Data.csv")
train = clean(raw)
train = train[train["client_category"] != "UNKNOWN"].reset_index(drop=True)
print(f"[1] {len(train)} enrollments, {train[TARGET].mean():.1%} dropped")


# =====================================================================================
# Step 2 - Feature engineering
# =====================================================================================
# What happens here:
#   - Counts that are almost always 0 become yes/no flags, because 0 -> 1 is the signal
#     (prior dropout: 37% drop rate at 0, 95% at 1 or more).
#   - Company_ID, Agent_ID and Origin_Country keep only values with enough rows to trust
#     (n >= 30, or >= 4% of rows for countries); the rest are grouped as OTHER.
#   - Every categorical column is one-hot encoded; missing numbers get the median.
#   - Category lists and medians are learned once here and reused unchanged for new data.
#
# Note: prepaid, non-refundable enrollments outside affiliated admission (9,695 rows) drop
# 100% of the time. They are kept: payment terms are known at sign-up, and leaving them out
# would hide 37% of all dropouts from the model.

def frequent(series, min_count):
    counts = series.value_counts()
    return [str(int(v)) for v in counts[counts >= min_count].index]


countries = train.loc[train["country"] != "MISSING", "country"].value_counts(normalize=True)
VOCAB = {
    "country":   list(countries[countries >= 0.04].index),
    "lab":       list(train["Requested_Lab_Config"].dropna().value_counts().head(6).index),
    "company":   frequent(train["Company_ID"], 30),
    "agent":     frequent(train["Agent_ID"], 30),
    "medians":   {c: train[c].median() for c in ["Registration_Days_Before", "Students_Count",
                                                 "Practical_Hours", "Daily_Tuition_Cost"]},
    "first_day": train["Course_Start_Date"].min(),
}


def keep_or_other(values, keep, extra=()):
    return pd.Categorical(values.where(values.isin(list(keep) + list(extra)), "OTHER"),
                          categories=list(keep) + ["OTHER"] + list(extra))


def id_group(ids, keep):
    return keep_or_other(ids.apply(lambda v: "OTHER" if pd.isna(v) else str(int(v))), keep)


def build_features(df, columns=None):
    med = VOCAB["medians"]
    numeric = pd.DataFrame({
        "had_prior_dropout":        (df["Prev_Course_Dropouts"] >= 1).astype(int),
        "attended_before":          (df["Prev_Course_Attended"] >= 1).astype(int),
        "registration_days_before": df["Registration_Days_Before"].fillna(med["Registration_Days_Before"]),
        "waiting_list_days":        df["Waiting_List_Days"],
        "has_waited":               (df["Waiting_List_Days"] > 0).astype(int),
        "has_support_ticket":       (df["Pre_Course_Supports_Tickets"] > 0).astype(int),
        "professionals_count":      df["Professionals_Count"],
        "students_count":           df["Students_Count"].fillna(med["Students_Count"]),
        "has_observers":            (df["Observers_Count"] > 0).astype(int),
        "practical_hours":          df["Practical_Hours"].fillna(med["Practical_Hours"]),
        "theory_hours":             df["Theory_Hours"],
        "daily_tuition_cost":       df["Daily_Tuition_Cost"].fillna(med["Daily_Tuition_Cost"]),
        "is_zero_cost":             (df["Daily_Tuition_Cost"] == 0).astype(int),
        "returning_client":         df["Returning_Client"],
        "has_company_id":           df["Company_ID"].notna().astype(int),
        "has_agent_id":             df["Agent_ID"].notna().astype(int),
        "course_start_days":        (df["Course_Start_Date"] - VOCAB["first_day"]).dt.days,
    })
    changes = np.select([df["Registration_Changes"] == 0, df["Registration_Changes"] == 1], ["0", "1"], "2+")
    categorical = pd.DataFrame({
        "registration_changes": pd.Categorical(changes, categories=["0", "1", "2+"]),
        "requested_lab_config": keep_or_other(df["Requested_Lab_Config"].fillna("MISSING"), VOCAB["lab"], ["MISSING"]),
        "origin_country":       keep_or_other(df["country"], VOCAB["country"], ["MISSING"]),
        "client_category":      df["client_category"],
        "enrollment_type":      df["enrollment_type"],
        "submission_source":    df["submission_source"],
        "catering_package":     df["catering_package"],
        "payment_terms":        df["payment_terms"],
        "company_id_group":     id_group(df["Company_ID"], VOCAB["company"]),
        "agent_id_group":       id_group(df["Agent_ID"], VOCAB["agent"]),
    }, index=df.index)
    X = pd.concat([numeric, pd.get_dummies(categorical, prefix_sep="__").astype(int)], axis=1)
    return X if columns is None else X.reindex(columns=columns, fill_value=0)


X, y = build_features(train), train[TARGET]
print(f"[2] feature matrix {X.shape[0]} x {X.shape[1]}, {int(X.isna().sum().sum())} missing values")


# =====================================================================================
# Step 3 - Training and tuning
# =====================================================================================
# What happens here:
#   - 20% of rows (stratified) are locked away as the holdout before any training.
#   - On the remaining 80%, each model's settings are searched at random and scored by
#     ROC-AUC averaged over the same 5 cross-validation folds, so models compete fairly.
#   - Train-fold AUC is kept next to CV AUC: a large gap would mean overfitting.

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODELS = {
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED))]),
        {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"]},
        20),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"n_estimators": randint(200, 500), "max_depth": [None, 8, 12, 16, 24],
         "min_samples_leaf": randint(1, 20), "max_features": ["sqrt", "log2", 0.3, 0.5]},
        20),
    "XGBoost": (
        XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                      random_state=SEED, n_jobs=1),
        {"max_depth": randint(3, 13), "learning_rate": loguniform(1e-2, 3e-1),
         "n_estimators": randint(300, 900), "subsample": uniform(0.6, 0.4),
         "colsample_bytree": uniform(0.6, 0.4), "reg_alpha": loguniform(1e-3, 1e1),
         "reg_lambda": loguniform(1e-3, 1e1)},
        30),
}

fitted, cv_rows = {}, []
for name, (model, space, n_iter) in MODELS.items():
    search = RandomizedSearchCV(model, space, n_iter=n_iter, scoring="roc_auc", cv=folds, n_jobs=-1,
                                random_state=SEED, return_train_score=True)
    search.fit(X_train, y_train)
    fitted[name] = search.best_estimator_
    i = search.best_index_
    cv_rows.append(dict(model=name, cv_auc=search.cv_results_["mean_test_score"][i],
                        cv_auc_std=search.cv_results_["std_test_score"][i],
                        train_auc=search.cv_results_["mean_train_score"][i]))
    print(f"[3] {name}: CV AUC {cv_rows[-1]['cv_auc']:.3f} | settings {search.best_params_}")

cv_table = pd.DataFrame(cv_rows).set_index("model")
cv_table["overfit_gap"] = cv_table["train_auc"] - cv_table["cv_auc"]
cv_table.round(4).to_csv(RESULTS / "cv_scores.csv")


# =====================================================================================
# Step 4 - Evaluation on the holdout (used once) and explanation
# =====================================================================================
# What happens here:
#   - Each tuned model predicts the untouched holdout; a probability >= 0.5 means "will drop".
#   - Recall matters most: a missed dropout means nobody calls the client, while a false
#     alarm only costs an unnecessary call.
#   - The best model by holdout AUC is explained with SHAP: which features push a
#     prediction towards "drop".

probs, rows = {}, []
for name, model in fitted.items():
    probs[name] = model.predict_proba(X_holdout)[:, 1]
    pred = (probs[name] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_holdout, pred).ravel()
    rows.append(dict(model=name, auc=roc_auc_score(y_holdout, probs[name]),
                     recall=recall_score(y_holdout, pred), precision=precision_score(y_holdout, pred),
                     f1=f1_score(y_holdout, pred), accuracy=accuracy_score(y_holdout, pred),
                     caught=tp, missed=fn, false_alarms=fp, correctly_cleared=tn))
holdout = pd.DataFrame(rows).set_index("model")
holdout.round(4).to_csv(RESULTS / "holdout_scores.csv")
print("[4] holdout results:\n", holdout.round(3))

BEST = holdout["auc"].idxmax()
print(f"[4] best model: {BEST}")

# Figure: AUC on the data each model trained on, in cross-validation, and on the holdout.
# Similar CV and holdout bars mean the score generalises; a much taller train bar means overfitting.
stages = {"train folds": cv_table["train_auc"], "cross-validation": cv_table["cv_auc"], "holdout": holdout["auc"]}
fig, ax = plt.subplots(figsize=(8, 3.6))
for k, name in enumerate(fitted):
    xs = np.arange(len(stages)) + (k - 1) * 0.26
    bars = ax.bar(xs, [v[name] for v in stages.values()], 0.23, color=COLORS[name], label=name)
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, color=INK, padding=2)
ax.set_xticks(np.arange(len(stages)), list(stages))
ax.set_ylim(0.85, 1.0); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.tick_params(length=0)
ax.set_title("ROC-AUC at each stage", fontsize=11, color=INK, loc="left", pad=28)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.savefig(FIGURES / "generalisation.png"); plt.close(fig)

# Figure: AUC, recall and precision for each model on the holdout.
metrics = ["auc", "recall", "precision"]
fig, ax = plt.subplots(figsize=(8, 3.6))
width = 0.26
for k, name in enumerate(fitted):
    xs = np.arange(len(metrics)) + (k - 1) * width
    bars = ax.bar(xs, holdout.loc[name, metrics], width - 0.03, color=COLORS[name], label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=9, color=INK, padding=2)
ax.set_xticks(np.arange(len(metrics)), ["AUC", "Recall\n(dropouts caught)", "Precision\n(flags that are right)"])
ax.set_ylim(0.5, 1.02); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.tick_params(length=0)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.savefig(FIGURES / "model_comparison.png"); plt.close(fig)

# Figure: ROC curves - how well each model ranks dropouts above non-dropouts at every threshold.
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for name in fitted:
    fpr, tpr, _ = roc_curve(y_holdout, probs[name])
    ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name}  (AUC {holdout.loc[name, 'auc']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRID, label="random guess")
ax.set_xlabel("false alarm rate"); ax.set_ylabel("dropouts caught")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.savefig(FIGURES / "roc.png"); plt.close(fig)

# Figure: confusion matrix of the best model.
cells = holdout.loc[BEST, ["correctly_cleared", "false_alarms", "missed", "caught"]].astype(int).to_numpy().reshape(2, 2)
labels = [["correctly\ncleared", "false\nalarm"], ["missed", "caught"]]
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.imshow([[1, 0], [0, 1]], cmap=matplotlib.colors.ListedColormap(["#f1f0ec", "#cfe0f6"]))
for r in range(2):
    for c in range(2):
        ax.text(c, r, f"{cells[r, c]:,}\n{labels[r][c]}", ha="center", va="center", fontsize=11, color=INK)
ax.set_xticks([0, 1], ["predicted: stays", "predicted: drops"])
ax.set_yticks([0, 1], ["actually\nstayed", "actually\ndropped"])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title(f"{BEST} on {len(y_holdout):,} held-out enrollments", fontsize=11, color=INK)
fig.savefig(FIGURES / "confusion_matrix.png"); plt.close(fig)

# Figure: SHAP - average impact of the top features on the best model's predictions.
explainer = shap.TreeExplainer(fitted[BEST], X_train.sample(500, random_state=SEED))
shap_values = explainer(X_holdout.sample(2000, random_state=SEED))
importance = (pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)
                .sort_values(ascending=False).head(10)[::-1])
importance.round(4).to_csv(RESULTS / "shap_top_features.csv")
def pretty(feature):
    column, _, value = feature.partition("__")
    column = column.replace("_id_group", "").replace("_", " ").capitalize()
    if not value:
        return column
    return f"{column}: {value if len(value) <= 3 else value.lower()}"


pretty = [pretty(f) for f in importance.index]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(pretty, importance.values, color=COLORS[BEST], height=0.6)
ax.set_xlabel("average impact on prediction (mean |SHAP|)"); ax.tick_params(axis="y", length=0)
ax.set_title(f"What {BEST} looks at most", fontsize=11, color=INK, loc="left")
fig.savefig(FIGURES / "shap.png"); plt.close(fig)


# =====================================================================================
# Step 5 - Score new enrollments
# =====================================================================================
# The best model is refit with the same settings on all labelled rows, then scores the
# unlabelled file using the category lists and medians learned in step 2.

test_path = DATA_DIR / "Test_Data_No_Target.csv"
if test_path.exists():
    final_model = clone(fitted[BEST]).fit(X, y)
    new = clean(pd.read_csv(test_path))
    scores = final_model.predict_proba(build_features(new, columns=X.columns))[:, 1]
    pd.DataFrame({"Client_ID": new["Client_ID"], "Drop_Probability": scores}) \
      .to_csv(RESULTS / "drop_probabilities.csv", index=False)
    print(f"[5] scored {len(new)} new enrollments -> results/drop_probabilities.csv")
