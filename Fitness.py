# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load the dataset
df = pd.read_csv("gym_members_exercise_tracking_synthetic_data.csv")

# Handling Missing Values
# Fill numeric columns with the median and categorical columns with the mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Converting 'Max_BPM' to numeric (if needed)
df['Max_BPM'] = pd.to_numeric(df['Max_BPM'], errors='coerce').fillna(df['Max_BPM'].median())

# Encoding Categorical Variables
df = pd.get_dummies(df, columns=['Gender', 'Workout_Type'], drop_first=True)

# Check data types and missing values
df.info()
print("Missing Values Handled and Data Types Verified")

# Save cleaned data for visualization and modeling
cleaned_data_path = "cleaned_fitness_data.csv"
df.to_csv(cleaned_data_path, index=False)

# Exploratory Data Analysis (EDA)
# Plotting histograms for numerical features
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Boxplot for identifying outliers
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col, color='green')
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatterplots for key relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Session_Duration (hours)', y='Calories_Burned', hue='Experience_Level', palette='viridis')
plt.title("Session Duration vs Calories Burned")
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories Burned")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='BMI', y='Fat_Percentage', hue='Gender_Male', palette='coolwarm')
plt.title("BMI vs Fat Percentage")
plt.xlabel("BMI")
plt.ylabel("Fat Percentage")
plt.show()

# Pairplot for selected features
selected_features = ['Age', 'BMI', 'Calories_Burned', 'Workout_Frequency (days/week)']
sns.pairplot(df[selected_features])
plt.show()

# Feature Engineering
# Creating new feature: Weight-to-Height Ratio
df['Weight_to_Height_Ratio'] = df['Weight (kg)'] / df['Height (m)']

# Normalizing numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 'Calories_Burned',
                    'Session_Duration (hours)', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
                    'Experience_Level', 'Weight_to_Height_Ratio']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Model Building
# Defining features and target variable
X = df.drop(columns=['Calories_Burned'])
y = df['Calories_Burned']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Model Evaluation
print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test, lr_y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_y_pred):.4f}")

print("\nRandom Forest Regressor:")
print(f"MSE: {mean_squared_error(y_test, rf_y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_y_pred):.4f}")

# Streamlit Deployment
# Creating the Streamlit App
st.title("Fitness Tracker System")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", int(df['Age'].min() * 100), int(df['Age'].max() * 100), int(df['Age'].mean() * 100)) / 100
    weight = st.sidebar.slider("Weight (kg)", int(df['Weight (kg)'].min() * 100), int(df['Weight (kg)'].max() * 100), int(df['Weight (kg)'].mean() * 100)) / 100
    height = st.sidebar.slider("Height (m)", int(df['Height (m)'].min() * 100), int(df['Height (m)'].max() * 100), int(df['Height (m)'].mean() * 100)) / 100
    bmi = weight / (height ** 2)
    session_duration = st.sidebar.slider("Session Duration (hours)", int(df['Session_Duration (hours)'].min() * 100), int(df['Session_Duration (hours)'].max() * 100), int(df['Session_Duration (hours)'].mean() * 100)) / 100
    workout_frequency = st.sidebar.slider("Workout Frequency (days/week)", 0, 7, 3)
    data = {
        "Age": age,
        "Weight (kg)": weight,
        "Height (m)": height,
        "BMI": bmi,
        "Session_Duration (hours)": session_duration,
        "Workout_Frequency (days/week)": workout_frequency
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user inputs with normalized data
input_df_normalized = scaler.transform(input_df)

# Predict calories burned using Random Forest
prediction = rf_model.predict(input_df_normalized)
st.subheader("Predicted Calories Burned")
st.write(f"{prediction[0]:.2f} calories")
