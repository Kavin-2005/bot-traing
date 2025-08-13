import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1️⃣ Load the CSV file
file_path = r"C:\Users\CSE BDA\AppData\Local\Programs\Python\Python311\Salary_Data.csv"
dataset = pd.read_csv(file_path)

# 2️⃣ Rename columns for easier access
dataset.rename(columns={"Year of Experience": "Experience", "Salary": "Salary"}, inplace=True)

# 3️⃣ Remove missing values if any
dataset.dropna(inplace=True)

# 4️⃣ Features (X) and Labels (y)
X = dataset[["Experience"]]
y = dataset["Salary"]

# 5️⃣ Split into Training and Test Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 6️⃣ Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Predict on Training and Test Data
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 8️⃣ Model Performance Metrics
r2_train = r2_score(y_train, train_pred)
r2_test = r2_score(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
rmse = np.sqrt(mse)

# 9️⃣ Display Predictions
print("\n📊 Training Data Predictions:")
print(pd.DataFrame({
    "Experience": X_train["Experience"],
    "Actual Salary": y_train,
    "Predicted Salary": np.round(train_pred, 2)
}))

print("\n📊 Test Data Predictions:")
print(pd.DataFrame({
    "Experience": X_test["Experience"],
    "Actual Salary": y_test,
    "Predicted Salary": np.round(test_pred, 2)
}))

# 🔟 Display Model Performance
print("\n📈 Model Performance Metrics:")
print(f"R² Score (Train): {r2_train:.4f}")
print(f"R² Score (Test): {r2_test:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 1️⃣1️⃣ Plotting Training and Test Data
plt.figure(figsize=(8, 6))

# Sort values for smooth regression line
X_sorted = X_train.sort_values(by="Experience")
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(X_sorted, model.predict(X_sorted), color="red", linewidth=2, label="Regression Line")

plt.title("Experience vs Salary Prediction")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# 1️⃣2️⃣ Plot Actual vs Predicted for Test Data
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary (Test Data)")
plt.grid(True)
plt.show()
