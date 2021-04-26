import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn imports:
import sklearn
from sklearn import tree
from sklearn.metrics import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

PLOT_FONT_SIZE = 12    #font size for axis of plots

#define helper function for confusion matrix

def displayConfusionMatrix(confusionMatrix):
    """Confusion matrix plot"""
    
    confusionMatrix = np.transpose(confusionMatrix)
    
    ## calculate class level precision and recall from confusion matrix
    precisionLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1]))*100, 1)
    precisionHigh = round((confusionMatrix[1][1] / (confusionMatrix[1][0] + confusionMatrix[1][1]))*100, 1)
    recallLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))*100, 1)
    recallHigh = round((confusionMatrix[1][1] / (confusionMatrix[0][1] + confusionMatrix[1][1]))*100, 1)

    ## show heatmap
    plt.imshow(confusionMatrix, interpolation='nearest',cmap=plt.cm.Blues,vmin=0, vmax=100)
    
    ## axis labeling
    xticks = np.array([-0.5, 0, 1,1.5])
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(xticks)
    plt.gca().set_xticklabels(["","Class no \n Recall=" + str(recallLow), "Class yes \n Recall=" + str(recallHigh), ""], fontsize=PLOT_FONT_SIZE)
    plt.gca().set_yticklabels(["","Class no \n Precision=" + str(precisionLow), "Class yes \n Precision=" + str(precisionHigh), ""], fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Predicted Class", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Actual Class", fontsize=PLOT_FONT_SIZE)
        
    ## add text in heatmap boxes
    addText(xticks, xticks, confusionMatrix)
    
def addText(xticks, yticks, results):
    """Add text in the plot"""
    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, results[i][j], ha="center", va="center", color="white", size=PLOT_FONT_SIZE) ### size here is the size of text inside a single box in the heatmap
    
def calculateMetricsAndPrint(predictions, predictionsProbabilities, actualLabels):
    predictionsProbabilities = [item[1] for item in predictionsProbabilities]
    
    accuracy = accuracy_score(actualLabels, predictions) * 100
    precisionNegative = precision_score(actualLabels, predictions, average = None)[0] * 100
    precisionPositive = precision_score(actualLabels, predictions, average = None)[1] * 100
    recallNegative = recall_score(actualLabels, predictions, average = None)[0] * 100
    recallPositive = recall_score(actualLabels, predictions, average = None)[1] * 100
    auc = roc_auc_score(actualLabels, predictionsProbabilities) * 100
    
    print("Accuracy: %.2f\nPrecisionNegative: %.2f\nPrecisionPositive: %.2f\nRecallNegative: %.2f\nRecallPositive: %.2f\nAUC Score: %.2f\n" % 
          (accuracy, precisionNegative, precisionPositive, recallNegative, recallPositive, auc))
    
def custom_plot_roc_curve(fpr, tpr, lr_auc):
    print('AUC Score = %.3f' % (lr_auc * 100))
    plt.rcParams['figure.figsize'] = [6,6]
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()