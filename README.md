# Company Default Prediction

A machine learning project that predicts whether a company will default, using its financial data. The model is built to support a credit analyst, not to replace one.

## Problem

A company is marked as a **default** when its net worth next year is zero or below. About 7% of the companies in the data are defaults, so the data is imbalanced. A model that always predicts "no default" would already look about 93% accurate, so accuracy is not used as the main metric.

## Data

- 3,541 companies, 52 columns of financial figures.
- Columns with too many missing values were removed, and the company id was dropped.
- `Networth_Next_Year` is used only to build the target and is removed from the features, so the model never sees the answer.
- The dataset is not included in this repository.

## Approach

1. Clean the column names and build the target.
2. Split into train (75%) and test (25%) with the same default rate in both.
3. Fill missing values with a KNN imputer, inside the model pipeline so that no information leaks from the test data.
4. Compare three models with 5-fold cross-validation on the training data:
   - Random Forest with SMOTE
   - Random Forest with class weights
   - Gradient boosting with class weights
5. Choose the best model by PR-AUC, which suits rare events. Random Forest with class weights won, and it is also simpler than SMOTE.
6. Choose the decision threshold with cross-validation on the training data only, instead of using the default 0.5.
7. Check the final model once on the test set.

The comparison table is in Section 5 of the notebook.

## Results on the test set

886 companies, of which 61 are defaults.

| Metric | Value |
|---|---|
| ROC-AUC | 0.941 |
| PR-AUC | 0.623 |
| Precision (default class) | 0.66 |
| Recall (default class) | 0.61 |
| F1 (default class) | 0.63 |

In simple words: the model catches about 6 in 10 real defaults, and about 2 in 3 flagged companies are real defaults.

## Limitations

- Only 61 defaults are in the test set, so the numbers can change a little with a different split.
- The model misses about 4 in 10 real defaults, so it should support human review and not make decisions alone.
- The score is a ranking score, not an exact probability, because class weights change the scale.
- The data comes from one source and one period, so the model may not work for other markets without retraining.

## Files

| File | Purpose |
|---|---|
| `Company_Default_Prediction.ipynb` | Full project, step by step, with results |
| `train_model.py` | Trains and saves the final model as a script |
| `improve_model.py` | Compares the three models |
| `requirements.txt` | Libraries needed |

## How to run

1. `pip install -r requirements.txt`
2. Put your dataset in the folder as `Company data.csv`.
3. Open the notebook and run all cells, or run `python train_model.py`.

## Related project

The trained model is used as a tool inside an AI agent: [Company Risk Assistant](https://github.com/shraddha28201/company-risk-agent)

## Tech

Python, pandas, scikit-learn, imbalanced-learn, matplotlib.

## Author
Shraddha
