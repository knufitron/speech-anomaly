#!/usr/bin/env python3
"""Meta-analysis of the batch run results."""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Identify your feature types
categorical_features = ['model', 'scaler', 'feature_group']
numeric_features = ['actor_zscore']

# Create a preprocessor to encode categories
preprocessor = ColumnTransformer(
	 transformers=[
		 ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
		 ('num', 'passthrough', numeric_features)
	 ])

# Define the model pipeline
# Using the key parameters discussed: n_estimators, max_depth, etc.
rf_pipeline = Pipeline(steps=[
	 ('preprocessor', preprocessor),
	 ('regressor', RandomForestRegressor(
		 n_estimators=100,
		 max_depth=None,
		 random_state=42,
		 n_jobs=-1
	 ))
 ])

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python meta_analysis.py <path to collected_results_wo_top20.tsv'>")
		sys.exit(1)

	# Read data
	df = pd.read_csv(sys.argv[1], sep='\t')

	# Prepare data
	X = df[['model', 'scaler', 'actor_zscore', 'feature_group']]
	y = df['mcc']

	# Train the model
	rf_pipeline.fit(X, y)

	# Get the feature names after encoding
	ohe_features = (rf_pipeline.named_steps['preprocessor']
					.named_transformers_['cat']
					.get_feature_names_out(['model', 'scaler', 'feature_group']))

	# Combine OHE (one-hot encoded) names with your original numeric column name
	all_feature_names = [*ohe_features, 'actor_zscore']

	# Extract importances from the regressor
	importances = rf_pipeline.named_steps['regressor'].feature_importances_

	# Create a sorted DataFrame for easy reading
	importance_df = pd.DataFrame({
		'Feature': all_feature_names,
		'Importance': importances
	}).sort_values(by='Importance', ascending=False)

	print(importance_df)