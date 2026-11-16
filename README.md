# Stroke Prediction using Logistic Regression

BTEC International Level 3 Diploma in IT - Unit 21: Introduction to Artificial Intelligence (AI)
Oshwal College | Client scenario: Nairobi Smart Solutions Ltd. | Teacher: Silas Wawire

## Project summary
A supervised machine learning project that predicts whether a patient is likely to have a stroke from routine health
and demographic information. The model is **decision support for clinicians, not a diagnosis tool**.

Because only about 5% of patients in the dataset had a stroke, a model that always predicts "no stroke" scores about
95% accuracy while finding no strokes at all. The project is therefore judged mainly on **recall** and **ROC-AUC**.

## Dataset
Healthcare Stroke Prediction Dataset (fedesoriano, Kaggle):
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

5,110 patient records, 11 input features and the target `stroke` (1 = stroke, 0 = no stroke).
Download `healthcare-dataset-stroke-data.csv` from Kaggle (free account needed) and place it next to the notebook.

## What the notebook does
| Part | Steps |
|------|-------|
| Task 2 | Load data, audit quality, remove `id` and one `Other` gender row, stratified 80/20 split, gender-grouped median imputation of missing BMI, one-hot encoding, standard scaling |
| Task 3 | Train baseline and class-weighted Logistic Regression, cross-validate `C`, evaluate (recall, precision, ROC-AUC, confusion matrices), inspect coefficients, predict for new patients, explore decision thresholds |

Key design choices: the data is split **before** imputing and scaling (no test-set leakage); class imbalance is handled
with `class_weight='balanced'` (cost-sensitive learning); Logistic Regression is used because its coefficients can be
explained to clinicians.

## How to run
1. Open `Stroke_Prediction_Notebook.ipynb` in Google Colab (File > Upload notebook) or Jupyter.
2. Run all cells. When asked, upload `healthcare-dataset-stroke-data.csv`.


Local install: `pip install -r requirements.txt`

## Files
| File | Description |
|------|-------------|
| `Stroke_Prediction_Notebook.ipynb` | Full pipeline (Tasks 2 and 3) |
| `requirements.txt` | Python libraries |
| `cleaned_train.csv`, `cleaned_test.csv` | Prepared (imputed, encoded, scaled) data produced by the notebook |
| `model_comparison.csv` | Test-set metrics for the three models |
| `stroke_model.joblib` | Saved final model and preprocessing objects |
| `task2_*.png`, `task3_*.png` | Figures produced by the notebook |

## Results
Final model: Logistic Regression, `class_weight='balanced'`, `C=0.4`.
Test-set recall:0.80 | ROC-AUC:0.839 | Precision:0.13

## Limitations
Single public dataset of unknown origin, only 249 stroke cases (about 50 in the test set), no external validation, and
class-weighted scores are not calibrated probabilities. The model must not be used for real clinical decisions.

## Author
Mohamed mzahim
