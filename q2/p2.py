# Part 2: Gaussian Distribution Parametric Estimator and Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================  Discriminant Function   ========================================
def disc_f(x):
    g1 = -0.5* np.log(2* np.pi) - np.log(np.sqrt(var1)) - ((x - mean1)**2)/2*var1 + np.log(prior_prob[1])
    g2 = -0.5* np.log(2* np.pi) - np.log(np.sqrt(var2)) - ((x - mean2)**2)/2*var2 + np.log(prior_prob[2])

    if g1>g2:
        return 1
    else:
        return 2

# ============================  Data Input and Splitting  ========================================
data = pd.read_csv("input_2.csv")       #       Contains 2000 data input
train,test =  train_test_split(data, test_size = 0.2, shuffle = False) # train and test contains the training and testing data now
data_length = len(data)
# ============================  Calculating Prior of Classes    ====================================
prior = train['class'].value_counts()
prior_prob = prior/len(train)
print('prob1: ',prior_prob[1])
print('prob2: ',prior_prob[2])
# ==============================    Estimator   =======================================================
# Need to estimate for mean and variance for all classes

#   Mean
class1 = train[train['class']==1]
class2 = train[train['class']==2]
mean1 = np.mean(class1['feature_value'])
mean2 = np.mean(class2['feature_value'])
print(mean1)
print(class2)
print(mean2)

#   Variance
#   Creating dif_squared column for each class:
class1['dif_sqr'] = (class1['feature_value'] - mean1)**2
class2['dif_sqr'] = (class2['feature_value'] - mean2)**2

#   Finding Variance for each class
var1 = np.mean(class1['dif_sqr'])
var2 = np.mean(class2['dif_sqr'])

# ================================  Prediction  ============================================================
test['prediction'] = test['feature_value'].apply(func=disc_f)

labels = [1,2]
conf_matrix = confusion_matrix(test['class'], test['prediction'],labels=[1,2])
#print(conf_matrix)
cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
print('Confusion Matrix:\n',cm,'\n',sep="  ")

recall = np.diag(cm) / np.sum(cm, axis = 1)
print('Recall:\n',recall,'\n', sep="")

precision = np.diag(cm)/ np.sum(cm, axis = 0)
print('Precision:\n',precision, '\n',sep="")

accuracy = np.sum(np.diag(cm)) / len(test)
print('Accuracy:\n',accuracy,'\n', sep="")

f1 = 2* (precision*recall)/(precision+recall)
print('f1 score for each class:\n',f1,'\n', sep='')

f1_avg = np.mean(f1)
print('average f1 score is:\n',f1_avg, sep='')

