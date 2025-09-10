# Drought Risk Prediction Using Rainfall and Soil Data

## Overview
This project predicts drought risk levels (0–4) using rainfall time series and soil attributes. A RandomForestClassifier is trained after preprocessing a date column into year, month, and day of year. By applying machine learning techniques, the system classifies regions into drought severity categories, helping in early risk detection and sustainable water management.

## Repository Structure
```
drought-risk-prediction/
├── data/                  # sample datasets only
├── notebooks/             # Jupyter notebooks for EDA & training
├── model/                 # trained model files
├── app.py                 # Streamlit web app
├── requirements.txt       # Python dependencies
├── .gitignore             # exclude big CSVs, cache, model checkpoints
├── README.md              # project overview + dataset usage
```

## Dataset
- Four CSVs are used: `soil_data.csv`, `train_timeseries.csv`, `test_timeseries.csv`, `validation_timeseries.csv`.
- GitHub size limits: only small samples (first ~500 rows) are included in `data/` as:
  - `sample_soil_data.csv`
  - `sample_train_timeseries.csv`
  - `sample_test_timeseries.csv`
  - `sample_validation_timeseries.csv`
- Full datasets are hosted externally (placeholder):From kaggle

## Installation
1. Create and activate a virtual environment (optional).
2. Install dependencies:
```
pip install -r requirements.txt
```

## Training
Use the notebook to train and save the model:
1. Open Jupyter and run the notebook:
```
jupyter notebook notebooks/train_random_forest.ipynb
```
2. The notebook will:
   - Load `data/sample_train_timeseries.csv` (and merge soil data if key available)
   - Preprocess the date into `year`, `month`, `dayofyear`
   - Train a `RandomForestClassifier` to predict `score_class`
   - Save the model to `model/random_forest_model.joblib`

## Web App
Run the Streamlit app:
```
streamlit run app.py
```
- Upload a CSV with the same feature columns used during training plus a date column.
- The app will predict drought risk levels (0–4), display a table, and show a summary chart.

Alternatively, if you build a Flask app route, run with:
```
python app.py
```
(Current implementation uses Streamlit.)

## Notes
- `.gitignore` excludes large CSVs and artifacts. Only sample data is tracked.
- If no model is present, the Streamlit app will prompt you to run the training notebook first.
- Ensure your inference CSV contains a date column and all required features. Non-numeric columns are encoded via category codes during preprocessing.
