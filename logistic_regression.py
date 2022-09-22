from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

source = "training_energy"

features = pd.read_csv(source + "_input.csv", sep=",").to_numpy()
labels = pd.read_csv(source + "_label.csv", sep=",").to_numpy()

X_train, X_validate, y_train, y_validate = train_test_split(features, labels, \
        test_size=0.1, random_state=1337)

y_train = y_train.ravel()
y_validate = y_validate.ravel()

reg = LogisticRegression(solver="liblinear")
reg.fit(X_train, y_train)

y_prediction = reg.predict(X_validate)

ax = plt.subplot()
cmatrix = confusion_matrix(y_validate, y_prediction)
sns.heatmap(cmatrix, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Predicted labels', fontsize=15)
ax.set_ylabel('True labels', fontsize=15)
ax.set_title('Confusion Matrix', fontsize=15)

accuracy = accuracy_score(y_validate, y_prediction)
print(f"Prediction accuracy: {100*accuracy:.2f}%")

plt.show()
