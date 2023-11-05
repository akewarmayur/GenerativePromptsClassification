import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd

df = pd.read_csv("status7.csv")
dd = df["gtClassification"].to_list()



# Example actual and predicted labels for multiclass classification
actual_labels = df["gtClassification"]
predicted_labels = df["classification"]
print(set(actual_labels))
print(set(predicted_labels))
# Create a confusion matrix
unique_labels = list(set(list(set(actual_labels)) + list(set(predicted_labels))))
print(unique_labels)
cm = confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)


# Function to plot the confusion matrix
# Function to plot the confusion matrix
# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot the confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=unique_labels)
plt.show()







