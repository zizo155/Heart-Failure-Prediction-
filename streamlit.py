import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
@st.cache_data
def load_data(file=None):
    if file is None:
        return pd.read_csv("heart.csv")
    else:
        return pd.read_csv(file)

# Sidebar: File upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data()

# Sidebar: Select section to display
section = st.sidebar.selectbox("Select Section", ["Data Overview", "Visualizations", "Outlier Handling", "Feature Engineering", "Modeling"])

# Data Overview
if section == "Data Overview":
    st.title("Heart Disease Dataset Overview")
    st.write("### Dataset Head")
    st.dataframe(df.head())
    st.write("### Dataset Info")
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    st.write("### Summary Statistics")
    st.write(df.describe())

# Visualizations
elif section == "Visualizations":
    st.title("Data Visualizations")

    # Distribution of Numerical Features
    st.write("### Distribution of Numerical Features")
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(feature)
    plt.tight_layout()
    st.pyplot(fig)

    # Distribution of Categorical Features
    st.write("### Distribution of Categorical Features")
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for i, feature in enumerate(categorical_features):
        sns.countplot(data=df, x=feature, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(feature)
        axes[i//2, i%2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    df_encoded = df.copy()
    for feature in categorical_features:
        df_encoded[feature] = pd.factorize(df_encoded[feature])[0]
    corr_matrix = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Outlier Handling
elif section == "Outlier Handling":
    st.title("Outlier Handling")
    st.write("### Boxplot of Numerical Features Before Outlier Removal")
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numerical_features], ax=ax)
    st.pyplot(fig)

    # Function to remove outliers
    def remove_outliers(df, feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    # Remove outliers
    for feature in numerical_features:
        df = remove_outliers(df, feature)

    st.write("### Boxplot of Numerical Features After Outlier Removal")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numerical_features], ax=ax)
    st.pyplot(fig)
    st.write(f"Shape after removing outliers: {df.shape}")

# Feature Engineering
elif section == "Feature Engineering":
    st.title("Feature Engineering")

    # Convert categorical variables to numerical
    st.write("### Encoding Categorical Variables")
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Label Encoding
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        df[f'{feature}_encoded'] = label_encoder.fit_transform(df[feature])

    # One-Hot Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[categorical_features])
    onehot_columns = onehot_encoder.get_feature_names_out(categorical_features)
    df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_columns)

    # Combine one-hot encoded features with original dataframe
    df = pd.concat([df, df_onehot], axis=1)

    st.write("### Final Encoded DataFrame")
    st.write(df.head())

    # Normalizing/Standardizing Numerical Features
    st.write("### Normalizing/Standardizing Numerical Features")
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    st.write("Numerical features after scaling:")
    st.write(df[numerical_features].head())

    # Feature Creation
    st.write("### Feature Creation")
    df['Age_BP'] = df['Age'] * df['RestingBP']
    df['Cholesterol_Ratio'] = df['Cholesterol'] / df['Cholesterol'].mean()
    st.write("### Feature Engineered DataFrame")
    st.write(df.head())

# Modeling
elif section == "Modeling":
    st.title("Modeling")

    # Preparing the data
    st.write("### Preparing Data for Modeling")
    
    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Check if 'HeartDisease' is present and handle accordingly
    if 'HeartDisease' in df.columns:
        if df['HeartDisease'].dtype == 'object':
            # If 'HeartDisease' is categorical, encode it separately
            y = pd.get_dummies(df['HeartDisease'], prefix='HeartDisease')['HeartDisease_1']
            X = df.drop('HeartDisease', axis=1)
        else:
            # If 'HeartDisease' is already numeric, use it as is
            y = df['HeartDisease']
            X = df.drop('HeartDisease', axis=1)
    else:
        st.error("'HeartDisease' column not found in the dataset. Please check your data.")
        st.stop()

    # One-hot encode remaining categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Update numeric_columns to include only columns present in X_train
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Scale only the numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    st.write(f"Train set size: {X_train.shape}")
    st.write(f"Test set size: {X_test.shape}")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Evaluate each model
    for name, model in models.items():
        st.write(f"### Evaluating {name}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        st.write(f"Cross-validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate and display metrics
        st.write(f"Test set ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        
        # Feature importance (for Random Forest)
        if name == "Random Forest":
            st.write("### Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
            plt.title("Top 10 Feature Importance - Random Forest")
            st.pyplot(fig)
    
    # Model Comparison
    st.write("### Model Comparison")
    model_comparison = pd.DataFrame({
        'Model': models.keys(),
        'ROC AUC': [roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]) for model in models.values()]
    })
    model_comparison = model_comparison.sort_values('ROC AUC', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='ROC AUC', y='Model', data=model_comparison, ax=ax)
    plt.title("Model Comparison - ROC AUC Scores")
    st.pyplot(fig)
    
    st.write("Model Comparison Table:")
    st.table(model_comparison.style.highlight_max(axis=0))