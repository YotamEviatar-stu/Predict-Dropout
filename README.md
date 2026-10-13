# Predict-Dropout

**Can we flag which enrollments will drop a course, early enough to intervene?**

```mermaid
flowchart LR
    A[63,464 enrollments<br/>29 columns] --> B[Clean<br/>fix corrupted text,<br/>sentinel values, leakage]
    B --> C[Engineer features<br/>156 columns]
    C --> D[Tune 3 models<br/>5-fold CV]
    D --> E[Holdout test<br/>10,754 rows]
```

**Which model wins?** Holdout AUC, where 1.0 is perfect:

```mermaid
xychart-beta
    x-axis ["Logistic Regression", "Random Forest", "XGBoost"]
    y-axis "Holdout AUC" 0.8 --> 0.95
    bar [0.855, 0.928, 0.933]
```

**XGBoost on the holdout's 3,321 real dropouts:**

```mermaid
pie showData
    "caught" : 2455
    "missed" : 866
```

Of the enrollments it flags, 82% really do drop. The strongest predictors are a previous dropout, open support tickets, and origin country. Origin country partly stands in for client industry, because Portuguese clients skew toward Big Tech and IT.

📄 Details: [`extra/EXECUTIVE_SUMMARY.md`](extra/EXECUTIVE_SUMMARY.md)

---
<sub>The camera-trap notebook in this repo has its own clean version: [Detect-Species](https://github.com/YotamEviatar-stu/Detect-Species).</sub>
