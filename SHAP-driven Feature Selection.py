# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier  
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Data Path
data = pd.read_csv('/content/fetal_health.csv')

# Display the first five rows to verify
print("First five rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print(f"Dataset Shape: {data.shape}")

# Check the data types of each column
print("\nData Types:")
print(data.dtypes)

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Plot the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='fetal_health', data=data, palette='viridis')
plt.title('Distribution of Fetal Health Status')
plt.xlabel('Fetal Health')
plt.ylabel('Count')
plt.show()

# Convert 'fetal_health' to integer
data['fetal_health'] = data['fetal_health'].astype(int)

# Verify the conversion
print("\nData Types After Conversion:")
print(data.dtypes)

# Mapping numerical classes to descriptive labels
health_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
data['fetal_health_label'] = data['fetal_health'].map(health_mapping)

# Display the updated DataFrame
print("\nDataset with Mapped Labels:")
print(data[['fetal_health', 'fetal_health_label']].head())

# Features (all columns except 'fetal_health' and 'fetal_health_label')
X = data.drop(['fetal_health', 'fetal_health_label'], axis=1)

# Target variable
y = data['fetal_health']

# Visualize the original class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title('Original Class Distribution')
plt.xlabel('Fetal Health')
plt.ylabel('Count')
plt.show()

# Split the data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display the shapes of the splits
print(f"\nTraining Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the resampled training data and transform
X_train_scaled = scaler.fit_transform(X_train_resampled)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

# Convert the scaled arrays back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train_resampled.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# Verify scaling by checking means and standard deviations
print("\nMean of Scaled Training Features (Should be ~0):")
print(X_train_scaled.mean())

print("\nStandard Deviation of Scaled Training Features (Should be ~1):")
print(X_train_scaled.std())

# Initialize the LightGBM Classifier
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)

# Train the model on the resampled and scaled training data
lgbm_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test set
y_pred = lgbm_classifier.predict(X_test_scaled)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological'])
print("\nClassification Report:")
print(class_report)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Initialize the SHAP TreeExplainer
explainer = shap.TreeExplainer(lgbm_classifier)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test_scaled)

# Verify the structure of shap_values
print(f"Type of shap_values: {type(shap_values)}")
print(f"Shape of shap_values: {shap_values.shape}")
print(f"Model classes: {lgbm_classifier.classes_}")

# Define class names for plotting
class_names = ['Normal', 'Suspect', 'Pathological']

# Loop through each class to generate SHAP summary plots
for i, class_name in enumerate(class_names):
    print(f"\nGenerating SHAP Summary Plot for class: {class_name}")

    # Extract SHAP values for the current class
    shap_values_class = shap_values[:, :, i]

    # Verify the shape
    print(f"Shape of shap_values_class for {class_name}: {shap_values_class.shape}")

    # Generate the SHAP summary plot
    shap.summary_plot(
        shap_values_class,
        X_test_scaled,
        feature_names=X.columns,
        show=False
    )

    # Set the title for the plot
    plt.title(f'SHAP Summary Plot for {class_name}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
