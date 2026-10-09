
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create the machine learning model
model = DecisionTreeClassifier(random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict the species for test data
y_pred = model.predict(X_test)

# 6. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("Total dataset samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Flower Species:", iris.target_names[prediction[0]])
