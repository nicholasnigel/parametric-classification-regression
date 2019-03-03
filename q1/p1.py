# PROBLEM 1(CLASSIFICATION FROM Bernoulli Distributed Data)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ----------------------------------------  Discriminant Function   ----------------------------------------------------

def discriminant_function(x):
    pxc1 = (p1**(x)) * (1-p1)**(1-x)
    pxc2 = (p2**(x)) * (1-p2)**(1-x)

    g1 = pxc1 * prior_prob[1]
    g2 = pxc2 * prior_prob[2]


    print( "discriminant f 1: %f, discriminant f 2: %f "%(g1,g2) )
    
    # compare g1 and g2
    if g1>g2:
        return 1
    else:
        return 2





data = pd.read_csv("input_1.csv")   # Reading CSV file into 'data'
# print(data)
input_num = len(data)   #       length of input

# ----------------------------------------  Splitting data  ------------------------------------------------------------
train,test =  train_test_split(data, test_size = 0.2, shuffle = False) 

#print(train)
# ----------------------------------------  CALCULATING PRIOR PROBABILITY   --------------------------------------------

prior = train['class'].value_counts()    #       finding the value counts for each class
prior_prob = prior/input_num     #       Normalize by number of datas, contains prob[1] and prob[2]

# ----------------------------------------  Details  ------------------------------------------------------------
print(prior_prob)
print(
    "feature_value=1 , class=1: %d"%(train[ (train['feature_value'] == 1) & (train['class']==1)].shape[0])
)
print(
    "feature_value=0 , class=1: %d"%(train[ (train['feature_value'] == 0) & (train['class']==1)].shape[0])
)
print(
    "feature_value=1 , class=2: %d"%(train[ (train['feature_value'] == 1) & (train['class']==2)].shape[0])
)
print(
    "feature_value=0 , class=2: %d"%(train[ (train['feature_value'] == 0) & (train['class']==2)].shape[0])
)

# ----------------------------------------  Estimator   -----------------------------------------------------------------
# find p(x = 1|C1) and p(x = 1| C2) because p(x = 0 |Cn) = 1 - p(x = 1 | Cn) where x is the feature value

#print(data[ (data['feature_value']==1) & (data['class']==1) ].shape[0])

p1 = train[ (train['feature_value']==1) & (train['class']==1) ].shape[0]/ prior[1]     
p2 = train[ (train['feature_value']==1) & (train['class']==2) ].shape[0]/ prior[2]
print("p1: ", p1, " p2: ", p2)
test['prediction'] = -1 

test['prediction'] = test['feature_value'].apply(func=discriminant_function)

print(test)

# ==================================== Evaluating Matrics   ================================================================
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

