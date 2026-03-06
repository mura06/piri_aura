import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
from feature_enegineering import new_features

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.set_index('id')  # Set 'id' as the index column
print(df.head())
df = new_features(df)


X_train, X_val, y_train, y_val = train_test_split(
    df.drop(columns=['stroke']),
    df['stroke'],   
    test_size=0.2,
    stratify=df['stroke'],
    random_state=42
)

models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression()
}

plt.figure(figsize=(10, 6))

for name,model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Baseline
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Model Comparison for Stroke Prediction')
plt.legend()
plt.show()