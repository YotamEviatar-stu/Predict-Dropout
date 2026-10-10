# Predict-Dropout

Two machine-learning projects.

| | Project | Data | Model |
|---|---|---|---|
| 🎓 | [Course-dropout prediction](#-course-dropout-prediction) | 63,464 enrollments | XGBoost |
| 🦌 | [Camera-trap species detection](#-camera-trap-species-detection) | 2,500 labelled crops + 2,423 field frames | YOLO + MegaDetector |

---

## 🎓 Course-dropout prediction

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

Of the enrollments it flags, 82% really do drop. The strongest predictors are a previous dropout, origin country, and open support tickets.

📄 Details: [`extra/EXECUTIVE_SUMMARY.md`](extra/EXECUTIVE_SUMMARY.md)

---

## 🦌 Camera-trap species detection

**Which animals pass the camera traps, and do jackals outnumber gazelles?**

![one example per species](assets/species.jpg)

```mermaid
flowchart LR
    A[📷 Field frames] --> B[MegaDetector<br/>finds animals] --> C[YOLO classifier<br/>8 species] --> D{Confident?}
    D -- yes --> E[Species count]
    D -- no --> F[uncertain]
```

The model gets 96.6% of test images right and never confuses gazelle with jackal. In the May 2022 field data, jackal (18 crops) outnumbers gazelle (13).

📓 Notebook: [`Yotam_Eviatar_final_assignment.ipynb`](Yotam_Eviatar_final_assignment.ipynb) · 🧹 Clean version: [Detect-Species](https://github.com/YotamEviatar-stu/Detect-Species)
