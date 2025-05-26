# Survival Analysis with Kaplan-Meier Estimator

This project is part of the **AI for Medical Prognosis** specialization on Coursera by DeepLearning.AI. It explores statistical modeling techniques for survival data, with a focus on Kaplan-Meier estimation. The work was completed as the third weekly assignment in Course 2 of the specialization.

## 📘 Project Overview

We analyze a dataset of lymphoma patients and explore how to estimate survival probabilities, account for censored data, and compare survival curves across different patient subgroups.

### Key Concepts Covered

- Censored data and its interpretation in survival analysis
- Naive survival estimator vs. Kaplan-Meier estimator
- Subgroup survival curve analysis (Stage III vs. Stage IV cancer)
- Log-rank test for statistical significance

## 🧪 Technologies & Libraries

- Python
- Jupyter Notebook
- `lifelines` (for survival analysis)
- `pandas`, `numpy` (data manipulation)
- `matplotlib` (visualization)

## 📊 Highlights

- Calculated proportion of censored cases
- Built naive and Kaplan-Meier survival estimators from scratch
- Visualized survival curves and compared them across groups
- Applied log-rank test to assess statistical differences in survival

## 📈 Sample Output

**Survival Probability Estimates (90–360 days):**

| Days | Stage III | Stage IV |
|------|-----------|----------|
|  90  | 0.74      | 0.42     |
| 180  | 0.68      | 0.25     |
| 270  | 0.52      | 0.20     |
| 360  | 0.52      | 0.20     |

**Log-Rank Test p-value:** `0.0096`  
_Indicates a statistically significant difference in survival between groups._

## What I Learned
This project deepened my understanding of:

Survival modeling

How censored data affects analysis

Building models and comparing groups statistically
