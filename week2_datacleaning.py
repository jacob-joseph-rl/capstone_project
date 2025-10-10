import pandas as pd
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# Step 1: Fetch dataset
dataset = fetch_ucirepo(id=296)

# Step 2: Extract features (X) and targets (y)
X = dataset.data.features
y = dataset.data.targets

# Step 3: Identify columns for missing value imputation
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Step 4: Impute missing values
median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_cols] = median_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = mode_imputer.fit_transform(X[categorical_cols])

# Step 5: Remove or anonymize sensitive identifiers
sensitive_columns = ['patient_nbr']
X.drop(columns=[col for col in sensitive_columns if col in X.columns], inplace=True)

# Step 6: Standardize units if applicable (example placeholder)
# if 'glucose_mg_dL' in X.columns:
#     X['glucose_mmol_L'] = X['glucose_mg_dL'] * 0.0555

# Step 7: Create data dictionary using variable metadata descriptions
data_dictionary = pd.DataFrame({
    'Variable': X.columns,
    'Type': X.dtypes.astype(str),
    'Description': [dataset.variables.get(var, {}).get('description', '') for var in X.columns]
})

# Step 8: Lineage documentation for transparency
lineage_documentation = {
    'source_dataset_id': 296,
    'source': 'UCI ML Repository - Diabetes 130 US Hospitals 1999-2008',
    'cleaning_steps': [
        'Imputed missing numeric values with median',
        'Imputed missing categorical values with mode',
        'Removed sensitive patient identifiers',
        'Standardized units where applicable'
    ]
}

# Save cleaned dataframe to CSV file
X.to_csv('cleaned_data.csv', index=False)

# Output results
print("Cleaned Data Sample:")
print(X.head(10))

print("\nData Dictionary:")
print(data_dictionary)

print("\nLineage Documentation:")
print(lineage_documentation)
