import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# 7️⃣ Predict on Training Data
train_pred = model.predict(X_train)

# 8️⃣ Predict on Test Data
test_pred = model.predict(X_test)

# 9️⃣ Show Results
print("Training Data Predictions:")
print(pd.DataFrame({"Experience": X_train["Experience"], "Actual Salary": y_train, "Predicted Salary": train_pred}))

print("\nTest Data Predictions:")
print(pd.DataFrame({"Experience": X_test["Experience"], "Actual Salary": y_test, "Predicted Salary": test_pred}))

# 🔟 Plotting the graph
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(X_train, model.predict(X_train), color="red", label="Regression Line")
plt.title("Experience vs Salary Prediction")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
