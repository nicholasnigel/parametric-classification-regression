# Part 3: Multiple Class Gaussian Distribution 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from math import pow

# ================================  Discriminant Function   ============================================

def disc_f(x):
    g = {i:0 for i in train['class'].unique()}
    
    for i in train['class'].unique():
        g[i] = -1*np.log(np.sqrt(var_table[i]))
        g[i] = g[i] + np.log(prior_prob[i])
        u1 = pow(x-mean_table[i], 2)
        u1 = u1/(2*var_table[i])
        g[i] = g[i]-u1
    
    #       Find the Maximum of discirminant functions among those 4
    #print(g)
    return max(zip(g.values(), g.keys()))[1]

def split_data(data, training_size):
    """
    Splitting data into training and testing depending on training size
    
    Parameters
    ----------
        > data: The data/pandas dataframe you want to pass along
        > training_size: the proportion of data that is in training
    """
    num_training = int(training_size * len(data))
    num_test = len(data) - num_training
    train = data.head(num_training)
    test = data.tail(num_test)
    return train,test

def confusion_mat():
    """
    Creating confusion matrix that is useful for evaluation matrix

    Parameters
    ----------
        > actual: The list of actual classes
        > prediction: The list of predicted classes
    """
    classes = data['class'].unique()
    classes.sort()

    arr  = [[0 for i in range(len(classes))] for i in range(len(classes))]
    arr = np.array(arr)

    con_mat = pd.DataFrame(arr, index=classes, columns = classes)


    for i in classes:
        for j in classes:
            #con_mat.iloc[i][j] = test[ (test['class']==j ) & ( test['prediction'] == i) ]
            con_mat.loc[i,j] = len(test[ (test['class'] == i ) & ( test['prediction']==j ) ])
    return con_mat



# ================================  Data Input and Splitting    ========================================
f = open('detail3.txt', 'w')
f.write('PART 3, DATA DETAILS\n')
data = pd.read_csv('input_3.csv')
train,test =  split_data(data, 0.8)    #       splitting into training and data (80% and 20%)

# ================================  Compute Priors  ====================================================
prior = train['class'].value_counts()

prior_prob = prior/len(train)
f.write('Prior Probabilities of Each Class:\n'+str(prior_prob) + '\n\n')

# ================================  Estimator   ========================================================

# Making tables for all the specific classes
classes = {i:0 for i in train['class'].unique()}
for key in classes:
    classes[key] = train[train['class'] == key]


mean = {i:0 for i in train['class'].unique()}
for key in mean:
    mean[key] = np.mean(classes[key]['feature_value'])

mean_table = pd.Series(mean)
mean_table = mean_table.sort_index()
#print(mean_table)
f.write('Mean Table:\n'+str(mean_table)+ '\n\n')

for i in train['class'].unique():
    classes[i]['dif_sqr'] = (classes[i]['feature_value'] - mean[i])**2

variance = {i:0 for i in train['class'].unique()}
for key in variance:
    variance[key] = np.mean(classes[key]['dif_sqr'])

var_table = pd.Series(variance)
var_table = var_table.sort_index()
f.write('Variance Table:\n'+ str(var_table)+'\n\n')


# ================================  Prediction  ================================================================
test['prediction'] = test['feature_value'].apply(func=disc_f)
labels = [1,2,3,4]

#conf_matrix = confusion_matrix(test['class'], test['prediction'],labels=labels)

cm = confusion_mat()
f.write('Confusion Matrix:\n'+str(cm)+'\n\n')

recall = np.diag(cm) / np.sum(cm, axis = 1)
f.write('Recall:\n'+str(recall)+'\n\n')

precision = np.diag(cm)/ np.sum(cm, axis = 0)
f.write('Precision:\n'+str(precision)+'\n\n')

accuracy = np.sum(np.diag(cm)) / len(test)
f.write('Accuracy:\n'+str(accuracy)+'\n\n')
    

f1 = 2* (precision*recall)/(precision+recall)
f.write('f1 scores:\n'+str(f1)+'\n\n')

f1_avg = np.mean(f1)
f.write('Average f1 score: '+str(f1_avg)+'\n\n')