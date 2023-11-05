import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
"""
['violence' 'neutral' 'substance use' 'nudity' 'sexuality']

['weapons' 'neutral' 'blood and gore' 'prescription' 'hard drugs'
 'smoking, tobacco, marijuana' 'full' 'alcohol' 'intense' 'mild'
 'same sex' 'victim state' 'partial']
"""
df = pd.read_csv("status1.csv")
print(df.columns)

print(df["classification"].unique())
print(df["subclassification"].unique())

TP = len(df[df["status"] == "TP"])
FP = len(df[df["status"] == "FP"])
FN = len(df[df["status"] == "FN"])
TN = len(df[df["status"] == "TN"])

# Define the TP, FP, FN, and TN values


# Create the confusion matrix
confusion_matrix = [[TN, FP], [FN, TP]]

# Plot the confusion matrix
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Create tick marks for the classes
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])

# Add text annotations
thresh = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1])
plt.text(0, 0, f'TN = {confusion_matrix[0][0]}', horizontalalignment='center', verticalalignment='center', color='white' if confusion_matrix[0][0] > thresh else 'black')
plt.text(1, 0, f'FP = {confusion_matrix[0][1]}', horizontalalignment='center', verticalalignment='center', color='white' if confusion_matrix[0][1] > thresh else 'black')
plt.text(0, 1, f'FN = {confusion_matrix[1][0]}', horizontalalignment='center', verticalalignment='center', color='white' if confusion_matrix[1][0] > thresh else 'black')
plt.text(1, 1, f'TP = {confusion_matrix[1][1]}', horizontalalignment='center', verticalalignment='center', color='white' if confusion_matrix[1][1] > thresh else 'black')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

