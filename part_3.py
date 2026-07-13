import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv(r"C:\Users\saran\Documents\AIML\part_1\cleaned_data.csv")   # load the cleaned dataset
X = df.drop(columns=["final_score", "passed"]) # feature matrix
y_reg = df["final_score"]      # regression label         
y_clf = df["passed"]  # classification label
X["gender"] = X["gender"].map({"Male": 0, "Female": 1})  # encoding
X["internet_access"] = X["internet_access"].map({"Yes": 1, "No": 0})
X["extracurricular"] = X["extracurricular"].map({"Yes": 1, "No": 0})
y_clf = y_clf.map({"Yes": 1, "No": 0})
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42) # splitting the data
scaler = StandardScaler()   # scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
decision_tree_clf = DecisionTreeClassifier(random_state=42)   # training a decision tree classifier with no constraints
decision_tree_clf.fit(X_train_scaled, y_clf_train)
y_clf_train_pred = decision_tree_clf.predict(X_train_scaled)
y_clf_test_pred = decision_tree_clf.predict(X_test_scaled)
training_accuracy = accuracy_score(y_clf_train, y_clf_train_pred)  # training accuracy
test_accuracy = accuracy_score(y_clf_test, y_clf_test_pred)   # test accuracy
print("Decision Tree Training Accuracy: ", training_accuracy)
print("Decision Tree Test Accuracy: ", test_accuracy)
decision_tree_clf_limited = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)   # training a decision tree classifier with constraints
decision_tree_clf_limited.fit(X_train_scaled, y_clf_train)
y_clf_train_pred_limited = decision_tree_clf_limited.predict(X_train_scaled)
y_clf_test_pred_limited = decision_tree_clf_limited.predict(X_test_scaled)
training_accuracy_limited = accuracy_score(y_clf_train, y_clf_train_pred_limited)  # training accuracy of model with constraints
test_accuracy_limited = accuracy_score(y_clf_test, y_clf_test_pred_limited)   # test accuracy of model with constraints
print("Decision Tree Training Accuracy Limited: ", training_accuracy_limited)
print("Decision Tree Test Accuracy Limited: ", test_accuracy_limited)