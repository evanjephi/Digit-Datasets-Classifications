import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load the digits dataset
digit = load_digits()
data = np.c_[digit.data, digit.target]
columns = np.append(digit.feature_names, ["target"])
digit_df = pd.DataFrame(data, columns=columns)

# Assigning X with a callable to the sliced dataframe
X = digit_df.iloc[lambda x: x.index % 1 == 0]
y = digit_df.iloc[:, -1]

classifiers = ['DTC', 'KNNC', 'RFC', 'SVMC']
kf = KFold(n_splits=10)

for i in classifiers:
    arr_x_trainingData = []
    arr_x_testData = []
    arr_y_trainingData = []
    arr_y_testData = []
    arr_myAccScore = []

    if i == 'DTC':
        classifier = DecisionTreeClassifier()
    elif i == 'KNNC':
        classifier = KNeighborsClassifier()
    elif i == 'RFC':
        classifier = RandomForestClassifier()
    elif i == 'SVMC':
        classifier = SVC()

    for train_index, test_index in kf.split(X):
        X_trainingData, X_testData = X.iloc[train_index, :], X.iloc[test_index, :]
        y_trainingData, y_testData = y[train_index], y[test_index]
        classifier.fit(X_trainingData, y_trainingData)
        prediction = classifier.predict(X_testData)
        accuracy = accuracy_score(prediction, y_testData)

        arr_x_trainingData.append(X_trainingData)
        arr_x_testData.append(X_testData)
        arr_y_trainingData.append(y_trainingData)
        arr_y_testData.append(y_testData)
        arr_myAccScore.append(accuracy)

    maxAccuracy = np.argmax(arr_myAccScore)

    max_x_trainingData = arr_x_trainingData[maxAccuracy]
    max_x_testData = arr_x_testData[maxAccuracy]
    max_y_trainingData = arr_y_trainingData[maxAccuracy]
    max_y_testData = arr_y_testData[maxAccuracy]

    classifier.fit(max_x_trainingData, max_y_trainingData)
    prediction = classifier.predict(max_x_testData)

    print("{}:".format(i))
    print("Confusion Matrix")
    print(confusion_matrix(prediction, max_y_testData))
