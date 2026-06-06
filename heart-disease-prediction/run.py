import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'johnsmith88/heart-disease-dataset',
    'heart.csv'
)

# ====================== Start: EDA, Cleaning, Feature Analysis ======================
df = pd.DataFrame(df)
df.columns = df.columns.str.lower().str.replace('-', '_')

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(f'Rows: {len(df)}')
print(f'Duplicates: {df.duplicated().sum()}')

# canvas: distribution of target + feature histogram
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
axes = axes.flatten()

# distribution of target
df['target'].value_counts().plot(
    kind='bar',
    ax=axes[0],
    title='Distribution of target'
)

# feature histogram
for i, col in enumerate(['age', 'trestbps', 'thalach', 'chol', 'oldpeak']):
    axes[i + 1].hist(
        df[col].astype(float), 
        bins=25, 
        color='#534AB7', 
        edgecolor='white', 
        linewidth=0.4
    )
    axes[i + 1].set_title(col)

plt.tight_layout()
plt.savefig("images/eda_distributions.png", dpi=150)

# heatmap
corr = df.apply(pd.to_numeric, errors='coerce').corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
plt.tight_layout()
plt.savefig("images/eda_heatmap.png", dpi=150)

# Feature Significance vs target - Top 4 predictors by r
target = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
target = (target > 0).astype(int)

feature_significance = {}

for col in df.columns:
    if col == 'target': continue
    try:
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
        r, p = stats.pointbiserialr(target, vals)
        sig = 'SIGNIFICANT' if p < 0.05 else ''
        feature_significance[col] = r
    except Exception:
        pass

ranking = sorted(
    feature_significance.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)[:4]

print('\nFeature significance vs target')
for feature, r in ranking:
    print(f"{feature} r={r:+.3f}")

# ====================== End: EDA, Cleaning, Feature Analysis ======================

# ====================== Start: Baseline Models ======================

# handling null data, fix dtypes
df['target'] = (pd.to_numeric(
    df['target'], errors='coerce').fillna(0) > 0).astype(int)

for col in df.select_dtypes(include='object').columns:
  if col != 'target':
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

# We decide to remove duplicates in order to get the perfect prediction
df = df.drop_duplicates()

# processing models
FEATURE = [c for c in df.columns if c != 'target']
x = df[FEATURE]
y = df['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = joblib.load('model_tuned.pkl')

models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=42
    ),
    'XGBoost (optuna tuned)': model
}

model_cv_comparison = []

for i, (name, model) in enumerate(models.items()):
    auc = cross_val_score(model, x, y, cv=cv, scoring='roc_auc')
    f1 = cross_val_score(model, x, y, cv=cv, scoring='f1')
    model_cv_comparison.insert(i, {
        'name': name,
        'auc_mean': round(auc.mean(), 3),
        'auc_std': round(auc.std(), 3),
        'f1_mean': round(f1.mean(), 3),
        'f1_std': round(f1.std(), 3)
    })

model_cv_comparison = sorted(
    model_cv_comparison,
    key=lambda x: (
        x['auc_mean'],
        x['f1_mean'],
        -x['f1_std']
    ),
    reverse=True
)

formatted_result = [
    {
        'Model': result['name'],
        'AUC (5-fold CV)': f'{result['auc_mean']} ± {result['auc_std']}',
        'F1': f'{result['f1_mean']} ± {result['f1_std']}'
    }
    for result in model_cv_comparison
]
print(pd.DataFrame(formatted_result))

# Find Where Model Fails
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
best_model = {}

print('\nWhere Model Fails: \n')

failure_summary = []
for name, model in models.items():
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    failed_cases = x_test[y_test != y_pred].copy()
    failed_cases['actual'] = y_test[y_test != y_pred]
    failed_cases['predicted'] = y_pred[y_test != y_pred]

    fp = failed_cases[
        (failed_cases['actual'] == 0) &
        (failed_cases['predicted'] == 1)
    ]
    fn = failed_cases[
        (failed_cases['actual'] == 1) &
        (failed_cases['predicted'] == 0)
    ]
    failure_summary.append({
        'Model': name,
        'FP': len(fp),
        'FN': len(fn),
        'Total Errors': len(fp) + len(fn)
    })

    if name == 'XGBoost (optuna tuned)':
        # print(f'False Negative: \n{fn.head()}')
        # print(f'\nFalse Positive: \n{fp.head()}')
        # print(f'\nOriginal Dataset: \n{df[FEATURE].mean()}')
        feat = ['age','ca','cp','oldpeak']
        
        comparison = pd.DataFrame({
            'FN': round(fn[feat].mean(), 3),
            'FP': round(fp[feat].mean(), 3),
            'Dataset': round(df[feat].mean(), 3)
        })
        print(comparison)

print('\nFailure summary between 3 models:')
print(pd.DataFrame(failure_summary))

# Testing Tuned Model
def predict(input_dict: dict) -> dict:
  FEATS = list(input_dict.keys()) if input_dict else []

  df_in = pd.DataFrame([input_dict])
  for col in FEATS:
    if col not in df_in.columns:
      df_in[col] = 0

  df_in = df_in[FEATS]
  prob = model.predict_proba(df_in)[0, 1]
  pred = int(prob >= 0.5)
  risk = 'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'
  return {
      'prediction': pred,
      'probability': round(float(prob), 3),
      'risk_level': risk
  }

print('\nModel prediction after tunned:')
df = df.drop(columns=['target']).to_dict('records')

for i, ex in enumerate(df):
    if i < 10:
        print(predict(ex))