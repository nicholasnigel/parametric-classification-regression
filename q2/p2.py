# Part 2: Gaussian Distribution Parametric Estimator and Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from math import pow

# ============================  Discriminant Function   ========================================
def disc_f(x):
    #       Function meant to calculate p(x|Ci)

    g1 = -1*np.log(np.sqrt(var1))
    g2 = -1*np.log(np.sqrt(var2))
    #print(g1)
    #print(g2)
    #print('=========')
    g1 = g1 + np.log(prior_prob[1])
    g2 = g2 + np.log(prior_prob[2])
    #print(g1)
    #print(g2)
    #print('=========')
    u1 = pow(x-mean1, 2)
    u1 = u1/(2*var1)
    u2 = pow(x-mean2, 2)
    u2 = u2/(2*var2)
    #print(u1)
    #print(u2)
    #print('==========')
    g1 = g1-u1
    g2 = g2-u2
    #print(g1)
    #print(g2)
    if g1>g2:
        return 1
    else:
        return 2



# ============================  Data Input and Splitting  ========================================
f = open('det.txt', 'w')
data = pd.read_csv("input_2.csv")       #       Contains 2000 data input
train,test =  train_test_split(data, test_size = 0.2, shuffle = False) # train and test contains the training and testing data now
data_length = len(data)
# ============================  Calculating Prior of Classes    ====================================
prior = train['class'].value_counts()
prior_prob = prior/len(train)
print('prior: \n', prior, sep='')
print('prob1: ',prior_prob[1])
print('prob2: ',prior_prob[2])
f.write('prior prob 1 = '+str(prior_prob[1]) +'\n')
f.write('prior prob 2 = '+str(prior_prob[2])+ '\n')

# ==============================    Estimator   =======================================================
# Need to estimate for mean and variance for all classes

#   Mean
class1 = train[train['class']==1]
class2 = train[train['class']==2]

mean1 = np.mean(class1['feature_value'])
mean2 = np.mean(class2['feature_value'])

print("mean of class 1: ",mean1)
print("mean of class 2: ",mean2)
f.write('average of class 1 = '+str(mean1) + '\n')
f.write('average of class 2 = '+str(mean2) + '\n')


#   Variance
#   Creating dif_squared column for each class:
class1['dif_sqr'] = (class1['feature_value'] - mean1)**2
class2['dif_sqr'] = (class2['feature_value'] - mean2)**2


#   Finding Variance for each class
var1 = np.mean(class1['dif_sqr'])
var2 = np.mean(class2['dif_sqr'])

print("Variance of class1: ", var1)
print("Variance of class2: ", var2)

f.write("Variance of class 1 = "+ str(var1)+'\n')
f.write("Variance of class 2 = "+ str(var2)+'\n')




# ================================  Prediction  ============================================================
test['prediction'] = test['feature_value'].apply(func=disc_f)

labels = [1,2]
conf_matrix = confusion_matrix(test['class'], test['prediction'],labels=[1,2])
#print(conf_matrix)
cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
print('Confusion Matrix:\n',cm,'\n',sep="")
f.write('Confusion Matrix:\n'+str(cm)+'\n')

recall = np.diag(cm) / np.sum(cm, axis = 1)
print('Recall:\n',recall,'\n', sep="")
f.write('Recall:\n'+str(recall)+'\n')

precision = np.diag(cm)/ np.sum(cm, axis = 0)
print('Precision:\n',precision, '\n',sep="")
f.write('Precision:\n'+str(precision)+'\n')

accuracy = np.sum(np.diag(cm)) / len(test)
print('Accuracy:\n',accuracy,'\n', sep="")
f.write('Accuracy:\n'+str(accuracy)+'\n')
    

f1 = 2* (precision*recall)/(precision+recall)
print('f1 score for each class:\n',f1,'\n', sep='')
print('f1 scores:\n'+str(f1)+'\n')

f1_avg = np.mean(f1)
print('average f1 score is:\n',f1_avg, sep='')
f.write('Average f1 score: '+str(f1_avg)+'\n')

