from __future__ import annotations

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
	from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
	XGBClassifier = None  # type: ignore[misc, assignment]


MODEL_REGISTRY: dict[str, type] = {
	# ** Linear Models **
	# A linear model that is good for binary classification.
	# It is a good baseline model for binary classification because it is simple and fast to train.
	# Despite the name, this is a linear model used for classification. It uses a linear equation
	# to find the probability of a data point belonging to a specific category.
	"logistic_regression": LogisticRegression,

	# ** Ensemble & Non-Linear Models **
	# A non-linear model that is good for binary classification. It builds multiple decision trees and
	# combines them to make a prediction.
	# An ensemble of many Decision Trees. It uses "bagging" to average results and handle complex,
	# non-linear data patterns.
	"random_forest": RandomForestClassifier,
	# "xgb_classifier": XGBClassifier

	# ** Anomaly (Outlier) Detection Models **
	# A non-linear boundary model. It captures the "shape" of the normal data to identify anything
	# falling outside that boundary as an outlier.
	"one_class_svm": OneClassSVM,
	# An ensemble approach for anomalies. It "isolates" outliers by randomly partitioning the data;
	# outliers are easier to isolate (require fewer splits) than normal points.
	"isolation_forest": IsolationForest,
	# A density-based model. It compares the density of a point to its neighbors; if a point is in a much
	# sparser area than its neighbors, it's flagged.
	"local_outlier_factor": LocalOutlierFactor,
	# A linear/statistical approach for outlier detection. It assumes your "normal" data follows a Gaussian
	# (bell curve) distribution and draws an ellipse around it.
	"elliptic_envelope": EllipticEnvelope,
}

if XGBClassifier is not None:
	# XGBoost is a powerful gradient boosting library that can handle large datasets and complex models.
	# A high-performance ensemble model that builds trees sequentially, with each new tree correcting
	# the errors of the previous ones.
	MODEL_REGISTRY["xgb_classifier"] = XGBClassifier


SUPERVISED_MODELS: frozenset[str] = frozenset({"logistic_regression", "random_forest", "xgb_classifier"})
