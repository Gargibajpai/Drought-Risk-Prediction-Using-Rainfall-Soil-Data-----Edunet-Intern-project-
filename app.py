import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join('model', 'random_forest_model.joblib')
SAMPLE_DIR = 'data'

st.set_page_config(page_title='Drought Risk Prediction', layout='wide')
st.title('Drought Risk Prediction Using Rainfall and Soil Data')

st.markdown('Upload a CSV with a date column (e.g., date) and features used by the model. The app will predict drought risk levels (0–4).')

@st.cache_data
def _load_model_from_disk(path: str):
	if os.path.exists(path):
		try:
			return joblib.load(path)
		except Exception as e:
			st.warning(f'Failed to load model: {e}')
			return None
	return None

# Persist model across interactions
if 'model' not in st.session_state:
	st.session_state['model'] = _load_model_from_disk(MODEL_PATH)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	# Find a plausible date column
	date_col = None
	for cand in ['date', 'timestamp', 'time', 'Date', 'DATE']:
		if cand in df.columns:
			date_col = cand
			break
	if date_col is not None:
		df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
		df['year'] = df[date_col].dt.year
		df['month'] = df[date_col].dt.month
		df['dayofyear'] = df[date_col].dt.dayofyear
		df = df.drop(columns=[date_col])
	# Drop target if present
	if 'score_class' in df.columns:
		df = df.drop(columns=['score_class'])
	# Convert categorical columns to category codes
	for col in df.columns:
		if df[col].dtype == 'object':
			df[col] = df[col].astype('category').cat.codes
	# Fill NaNs
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.fillna(df.median(numeric_only=True))
	df = df.fillna(0)
	return df


def _first_existing_path(paths):
	for p in paths:
		if p and os.path.exists(p):
			return p
	return None


def _derive_score_class(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
	"""Try to derive score_class if missing using alternatives or rainfall heuristic. Returns (series, method)."""
	cols = set(df.columns)
	# 1) score -> qcut
	if 'score' in cols:
		try:
			bins = pd.qcut(df['score'], q=5, labels=False, duplicates='drop')
			return bins.astype(int), 'derived_from_score_qcut'
		except Exception:
			pass
	# 2) direct risk columns
	for cand in ['drought_risk', 'risk', 'risk_class', 'class']:
		if cand in cols:
			try:
				return df[cand].astype(float).round().clip(0, 4).astype(int), f'copied_from_{cand}'
			except Exception:
				pass
	# 3) Rainfall heuristic (lower rainfall → higher drought risk)
	for rain_col in ['PRECTOT', 'RAIN', 'PRCP', 'precip', 'precipitation']:
		if rain_col in cols:
			series = pd.to_numeric(df[rain_col], errors='coerce')
			try:
				inv = -series
				bins = pd.qcut(inv, q=5, labels=False, duplicates='drop')
				return bins.astype(int), f'heuristic_from_{rain_col}_qcut'
			except Exception:
				quantiles = series.quantile([0.2, 0.4, 0.6, 0.8]).values
				labels = [0,1,2,3,4]
				ser = pd.cut(series, bins=[-np.inf, *quantiles, np.inf], labels=labels, include_lowest=True)
				return ser.astype(int), f'heuristic_from_{rain_col}_fixed_bins'
	# 4) Last resort: use median of all numeric features to bin
	num_cols = df.select_dtypes(include=[np.number]).columns
	if len(num_cols) >= 1:
		mix = df[num_cols].median(axis=1)
		try:
			bins = pd.qcut(mix, q=5, labels=False, duplicates='drop')
			return bins.astype(int), 'fallback_from_numeric_median_qcut'
		except Exception:
			pass
	return None, ''


def _ensure_multiclass(y: pd.Series, source_df: pd.DataFrame) -> tuple[pd.Series | None, str]:
	"""If y has <2 classes, try fallback strategies to create at least 2 classes."""
	unique = np.unique(y.dropna())
	if len(unique) >= 2:
		return y, 'original'
	# Try binary median split on rainfall-like column
	for rain_col in ['PRECTOT', 'RAIN', 'PRCP', 'precip', 'precipitation']:
		if rain_col in source_df.columns:
			vals = pd.to_numeric(source_df[rain_col], errors='coerce')
			median = vals.median()
			labels = (vals > median).astype(int)
			if labels.nunique() >= 2:
				return labels, f'binary_split_{rain_col}_median'
	# Try rank-based buckets into 2 classes
	rank = source_df.select_dtypes(include=[np.number]).sum(axis=1).rank(method='first')
	labels = (rank > rank.median()).astype(int)
	if labels.nunique() >= 2:
		return labels, 'binary_split_numeric_rank'
	return None, 'failed'


def _fit_and_save(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
	clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
	clf.fit(X, y)
	os.makedirs('model', exist_ok=True)
	joblib.dump(clf, MODEL_PATH)
	st.session_state['model'] = clf
	return clf


def train_quick_demo_model() -> bool:
	"""Train a quick RandomForest model using available local data and save it."""
	try:
		train_candidates = [
			os.path.join(SAMPLE_DIR, 'sample_train_timeseries.csv'),
			os.path.join(SAMPLE_DIR, 'train_timeseries.csv'),
			' train_timeseries.csv'.strip()
		]
		soil_candidates = [
			os.path.join(SAMPLE_DIR, 'sample_soil_data.csv'),
			os.path.join(SAMPLE_DIR, 'soil_data.csv'),
			' soil_data.csv'.strip()
		]
		train_path = _first_existing_path(train_candidates)
		soil_path = _first_existing_path(soil_candidates)
		if not train_path:
			st.error('Training data not found. Expected one of: data/sample_train_timeseries.csv, data/train_timeseries.csv, train_timeseries.csv')
			return False
		st.info(f'Using training file: {train_path}')
		df_train = pd.read_csv(train_path, nrows=5000)
		df_soil = pd.read_csv(soil_path) if soil_path else None
		# Date features
		date_col = None
		for c in ['date','timestamp','time','Date','DATE']:
			if c in df_train.columns:
				date_col = c
				break
		if date_col is not None:
			df_train[date_col] = pd.to_datetime(df_train[date_col], errors='coerce')
			df_train['year'] = df_train[date_col].dt.year
			df_train['month'] = df_train[date_col].dt.month
			df_train['dayofyear'] = df_train[date_col].dt.dayofyear
			df_train = df_train.drop(columns=[date_col])
		# Optional merge
		merge_key = None
		if df_soil is not None:
			common = set(df_train.columns).intersection(set(df_soil.columns))
			for cand in ['location_id','station_id','site_id','region','grid_id']:
				if cand in common:
					merge_key = cand
					break
			if merge_key is not None:
				df_train = df_train.merge(df_soil, on=merge_key, how='left')
		if 'score_class' not in df_train.columns:
			derived, method = _derive_score_class(df_train)
			if derived is None:
				st.error('Could not derive target. Provide score_class or rainfall column.')
				return False
			st.success(f'Derived target via {method}.')
			df_train['score_class'] = derived
		y = df_train['score_class']
		X = df_train.drop(columns=['score_class'])
		for col in X.columns:
			if X[col].dtype == 'object':
				X[col] = X[col].astype('category').cat.codes
		X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0)
		# Ensure multiclass (>=2)
		y2, how = _ensure_multiclass(y, df_train)
		if y2 is None:
			st.error('Target still has one class after heuristics. Please supply more varied data.')
			return False
		if how != 'original':
			st.info(f'Target adjusted to two classes using {how}.')
		_fit_and_save(X, y2)
		st.success('Model saved and ready for predictions.')
		return True
	except Exception as e:
		st.error(f'Quick training failed: {e}')
		return False

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

# Controls
col1, col2, col3 = st.columns(3)
with col1:
	if st.button('Reload saved model'):
		st.session_state['model'] = _load_model_from_disk(MODEL_PATH)
		st.success('Reload attempted. If no file exists, model stays empty.')
with col2:
	if st.button('Force clear cache and rerun'):
		st.cache_data.clear()
		st.rerun()

if st.session_state['model'] is None:
	st.warning('No trained model found. Use quick training or train from uploaded CSV below.')
	with st.expander('Train a quick demo model from local files (recommended)'):
		if st.button('Train quick demo model'):
			with st.spinner('Training model...'):
				success = train_quick_demo_model()
				if success:
					st.success('Model trained and saved. You can now run predictions below.')

# Upload handling
user_df = None
if uploaded_file is not None:
	try:
		user_df = pd.read_csv(uploaded_file)
	except Exception:
		uploaded_file.seek(0)
		try:
			user_df = pd.read_csv(io.BytesIO(uploaded_file.read()))
		except Exception as e:
			st.error(f'Failed to read CSV: {e}')
			user_df = None

if user_df is not None:
	st.subheader('Preview')
	st.dataframe(user_df.head())
	X = preprocess_dataframe(user_df)
	st.subheader('Preprocessed Features')
	st.dataframe(X.head())

	# Auto-train from uploaded data if no model
	if st.session_state['model'] is None:
		train_df = user_df.copy()
		if 'score_class' not in train_df.columns:
			derived, method = _derive_score_class(train_df)
			if derived is not None:
				train_df['score_class'] = derived
				st.info(f'Derived target from uploaded CSV via {method}.')
		if 'score_class' in train_df.columns:
			Y = train_df['score_class']
			XX = preprocess_dataframe(train_df)
			Y2, how = _ensure_multiclass(Y, train_df)
			if Y2 is not None:
				_fit_and_save(XX, Y2)
				st.success('Model trained from uploaded CSV. Predictions enabled below.')
			else:
				st.error('Uploaded CSV target still has one class; need more varied data.')
		else:
			st.info('Uploaded CSV lacks label and could not derive one. Add rainfall column or a target.')

	# Predict if model exists
	if st.session_state['model'] is None:
		st.info('Train or provide a model to enable predictions.')
	else:
		try:
			preds = st.session_state['model'].predict(X)
			result_df = user_df.copy()
			result_df['predicted_score_class'] = preds
			st.subheader('Predictions')
			st.dataframe(result_df.head(100))

			counts = pd.Series(preds).value_counts().sort_index()
			st.subheader('Prediction Distribution')
			st.bar_chart(counts)

			csv = result_df.to_csv(index=False).encode('utf-8')
			st.download_button('Download Predictions CSV', csv, file_name='predictions.csv', mime='text/csv')
		except Exception as e:
			st.error(f'Prediction failed: {e}')
else:
	st.info('Upload a CSV to begin.')
