import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# === Connect and Load Data ===
engine = create_engine('postgresql://jacob:test123@localhost:5432/healthcare_db')
df = pd.read_sql('SELECT * FROM diabetes_hospital_data', engine)

# === Create Treatment Success Flag ===
median_stay = df['time_in_hospital'].median()
df['treatment_success'] = ((df['discharge_disposition_id'] == 1) & (df['time_in_hospital'] < median_stay)).astype(int)

# === Features for modeling ===
features = [
    'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty',
    'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'bmi', 'height'
]

def run_model(df, features, target_name, model_name='Logistic Regression'):
    print(f"\n\nMODELING FOR TARGET: {target_name}\n")
    # Prep dataframe
    df_ = df.dropna(subset=[target_name])
    X = pd.get_dummies(df_[features], drop_first=True)
    y = df_[target_name].astype(int)
    
    # Impute missing values (most frequent for any column)
    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # Train model
    clf = LogisticRegression(max_iter=500, solver='liblinear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    # Report
    print("\n==== Results ====")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # Feature importance
    importances = pd.Series(clf.coef_[0], index=X.columns)
    top_features = importances.abs().sort_values(ascending=False).head(10)
    print("\nTop 10 Predictors:")
    print(top_features)
    plt.figure(figsize=(8,4))
    top_features.plot(kind='barh')
    plt.title(f'Top 10 Features ({target_name})')
    plt.xlabel('Abs. Coefficient')
    plt.gca().invert_yaxis()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {target_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Random Forest for robustness
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)
    print("Random Forest Top Features:")
    print(rf_imp.sort_values(ascending=False).head(10))
    plt.figure(figsize=(8,4))
    rf_imp.sort_values(ascending=False).head(10).plot(kind='barh')
    plt.title(f'Random Forest Top 10 Features ({target_name})')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.show()

# === 1. Predict readmission risk ===
run_model(df, features, 'readmission_risk', 'Logistic Regression')

# === 2. Identify factors for treatment success ===
run_model(df, features, 'treatment_success', 'Logistic Regression')
