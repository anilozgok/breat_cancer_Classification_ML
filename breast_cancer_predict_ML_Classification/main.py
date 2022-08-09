import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# reading dataset
dataset = pd.read_csv("breast-cancer.csv")
# print(dataset.head())

# encoding diagnosis to numerical values for machine to learn
labelEncoder = LabelEncoder()
dataset["diagnosis"] = labelEncoder.fit_transform(dataset["diagnosis"].values)

# splitting dataset to train and test sets
train, test = train_test_split(dataset, test_size=0.3)

# creating X_train and y_train for model
X_train = train.drop("diagnosis", axis=1)
y_train = train.loc[:, "diagnosis"]

# creating X_test and y_test for model
X_test = test.drop("diagnosis", axis=1)
y_test = test.loc[:, "diagnosis"]

# LogisticRegression
model = LogisticRegression()
model = model.fit(X_train, y_train)

prediction = model.predict(X_test)
# print(prediction)

# confusion matrix
conMatrix = confusion_matrix(y_test, prediction)

# classification report
classificationReport = classification_report(y_test, prediction)
print(classificationReport)
