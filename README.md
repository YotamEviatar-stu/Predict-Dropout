# Predict-Dropout

**Which course enrollments will be dropped, and can we flag them early enough to call the client first?**

```mermaid
flowchart LR
    A[📋 63,462 enrollments<br/>41% dropped] --> B[Clean<br/>messy text, fake numbers,<br/>leaky columns]
    B --> C[156 features]
    C --> D[Tune 3 models<br/>5-fold CV]
    D --> E[Test once on<br/>12,693 held-out rows]
    E --> F[🎯 Drop probability<br/>for each new enrollment]
```

## What predicts a dropout

![drop rate by payment terms, prior dropout, support ticket, returning client](assets/drivers.png)

## Three problems the pipeline handles

**1. Messy text.** Category values were typed by hand, so one client type appears in 141 different spellings. Upper-casing and stripping symbols collapses them.

![141 spellings of one category collapse to one](assets/messy_text.png)

**2. Fake numbers.** Some values are impossible, such as 9,999 students or −5 practical hours. They mark missing data, so they're treated as missing and filled with the median.

**3. Columns that leak the answer.** Only data known at sign-up can be used. Some columns are filled in only after the client shows up, which gives the answer away.

```mermaid
flowchart LR
    A[Sign-up<br/>payment terms, history,<br/>country, agent] -->|✅ used| M[Model]
    B[Course starts<br/>course kit shipped,<br/>lab assigned] -.->|❌ dropped: leaks the answer| M
```

## Results

**Which model is best?** Holdout AUC (1.0 = perfect ranking):

```mermaid
xychart-beta
    x-axis ["Logistic Regression", "Random Forest", "XGBoost"]
    y-axis "Holdout AUC" 0.85 --> 1.0
    bar [0.909, 0.955, 0.957]
```

**XGBoost on the 5,260 real dropouts in the holdout:**

```mermaid
pie showData
    "caught" : 4383
    "missed" : 877
```

When it flags an enrollment, it's right 89% of the time (562 false alarms).

**What the model looks at:**

![top SHAP features](assets/shap.png)

Origin country PRT ranks high partly because it stands in for client type. Clients from PRT are mostly Big Tech and IT (66%), while other clients are mostly SaaS (66%).

---
Code: [`pipeline.py`](pipeline.py), five steps that read top to bottom. Run it with `pip install -r requirements.txt && python pipeline.py`. The data is in [`data/`](data/).
