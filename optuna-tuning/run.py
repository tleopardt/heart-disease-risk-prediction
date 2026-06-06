import optuna, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, classification_report
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'johnsmith88/heart-disease-dataset',
    'heart.csv'
)
# print(df.duplicated().sum()) - 723
# removing the duplicated data to avoid leakage data, identic sample, fair evaluation.
df = df.drop_duplicates()

FEATURE = [c for c in df.columns if c != 'target']
x = df[FEATURE]
y = df['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
  params = dict(
      n_estimators      = trial.suggest_int('n_estimators', 100, 500),
      max_depth         = trial.suggest_int('max_depth', 3, 7),
      learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
      subsample         = trial.suggest_float('subsample', 0.6, 1.0),
      colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
      reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
      reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 4.0),
      min_child_weight  = trial.suggest_int('min_child_weight', 1, 7),

      eval_metric='logloss',
      random_state=42
  )
  return cross_val_score(
      XGBClassifier(**params), x, y, cv=cv,
      scoring='roc_auc', n_jobs=-1
  ).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

best = XGBClassifier(**study.best_params, random_state=42)
best.fit(x, y)
joblib.dump(best, 'model_tuned.pkl')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
best.fit(x_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

RocCurveDisplay.from_estimator(best, x_test, y_test, ax=axes[0])
axes[0].plot([0, 1], [0, 1], '--', color='#B4B2A9', lw=1)
axes[0].set_title('ROC Curve')

PrecisionRecallDisplay.from_estimator(best, x_test, y_test, ax=axes[1], color='#534AB7')
axes[1].set_title('Precision-recall curve')

CalibrationDisplay.from_estimator(best, x_test, y_test, ax=axes[2], n_bins=8, color='#1D9E75')
axes[2].set_title('Calibration plot')

plt.tight_layout()
plt.savefig('images/eval_report.png', dpi=150)
# plt.show()

print(classification_report(y_test, best.predict(x_test)))

# feature importances
importances = pd.Series(best.feature_importances_, index=FEATURE)

fig, ax = plt.subplots(figsize=(7, 4))
importances.sort_values().plot(kind='barh', color='#B4B2A9', ax=ax)
ax.set_title('Feature Importances')
plt.tight_layout()
plt.savefig('images/heart-disease-feature-importances.png', dpi=150)
# plt.show()