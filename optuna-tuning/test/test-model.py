import joblib, pandas as pd, numpy as np

model = joblib.load('model_tuned.pkl')

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


examples = [
    {'age':63,'sex':1,'cp':3,'trestbps':145,'chol':233,'fbs':1,
     'restecg':0,'thalach':150,'exang':0,'oldpeak':2.3,'slope':0,
     'ca':0,'thal':1},
    {'age':37,'sex':1,'cp':2,'trestbps':130,'chol':250,'fbs':0,
     'restecg':1,'thalach':187,'exang':0,'oldpeak':3.5,'slope':0,
     'ca':0,'thal':2},
]

for ex in examples:
  print(predict(ex))