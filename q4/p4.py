# PART 4: Multi-feature Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from math import pow
from math import exp
from math import sqrt
from math import pi
# ================================
# Function bernoulli_likelihood:
# Evaluating the likelihood given that points follows bernoulli distribution
# Should evaluate the bernoulli of class and 1 and class 2
# ================================

def bernoulli_likelihood(x):
    bernoulli_class_1 = pow(p_1,x) * pow(1-p_1,1-x) 
    bernoulli_class_2 = pow(p_2,x) * pow(1-p_2,1-x)
    return pd.Series([bernoulli_class_1,bernoulli_class_2])



# ===================================
# Function gaussian_likelihood:
# Evaluating the likelihood given that points follows a normal(gaussian) distribution
# Should evaluate the estimate of normalized value
# ===================================
def gaussian_likelihood(x):
    gaussian_class_1 = 1/(sqrt(2*pi*variance_1))
    gaussian_class_2 = 1/(sqrt(2*pi*variance_2))

    temp_1 = -1* pow(x-mean_1,2) / (2*variance_1)
    temp_2 = -1* pow(x-mean_2,2) / (2*variance_2)

    gaussian_class_1 = gaussian_class_1 * exp(temp_1)
    gaussian_class_2 = gaussian_class_2 * exp(temp_2)

    return pd.Series([gaussian_class_1,gaussian_class_2])


def disc(r):
    g1 = r['b_1'] * r['gaus_1'] * prior_prob[1]
    g2 = r['b_2'] * r['gaus_2'] * prior_prob[2]
    if g1>g2:
        return 1
    else:
        return 2

# ====================================
# Function posterior:
# calculates posterior probability
# ====================================
def posterior():

    test[['b_1','b_2']]= test.apply(lambda row: bernoulli_likelihood(row['feature_value_1']),axis=1)
    test[['gaus_1','gaus_2']] = test.apply(lambda row: gaussian_likelihood(row['feature_value_2']),axis=1)
    test['prediction']=test.apply(disc,axis=1)
    print(test)

    #test['gaussian'] = test['feature_value_2'].apply(func=gaussian_likelihood).str.split(expand=True)
    #print(test)


# ============================================
# Main Function:
# ============================================



# ================================  Data Input and Splitting    ========================================
f = open('detail4.txt', 'w')
f.write('PART 3, DATA DETAILS\n')
data = pd.read_csv('input_4.csv')
train,test =  train_test_split(data, test_size = 0.2, shuffle = False)      #       splitting into training and data (80% and 20%)

# ================================  Calculating Prior Probability ========================================
prior = train['class'].value_counts()
prior_prob = prior/len(train)
print(prior_prob[1],prior_prob[2])
f.write('Prior Probabilities of Each Class:\n'+str(prior_prob) + '\n\n')
# ================================  Estimator for each Feature  ============================================
# Feature 1 is Bernoulli, 2 is Gaussian 

#       Separating into class1 and 2
class1 = train[train['class'] == 1]
class2 = train[train['class'] == 2]

# Estimating p1 and p2 for feature 1
p_1 = train[ (train['class']==1) & (train['feature_value_1']==1) ].shape[0]/len(class1)
p_2 = train[ (train['class']==2) & (train['feature_value_1']==1) ].shape[0]/len(class2)
f.write('Estimate Bernoulli p(i): \n')
f.write('p for class 1: '+str(p_1)+'\n')
f.write('p for class 2: '+str(p_2)+'\n')

#       Estimating Mean and variance of class 1 and 2
mean_1 = np.mean(class1['feature_value_2'])
mean_2 = np.mean(class2['feature_value_2'])

f.write('\nMean of feature 2 for each class:\n')
f.write('Class 1 mean: '+str(mean_1)+'\n')
f.write('Class 2 mean: '+str(mean_2)+'\n')


class1['dif_sqr'] = (class1['feature_value_2'] - mean_1)**2
class2['dif_sqr'] = (class2['feature_value_2'] - mean_2)**2

#       Estimating variances
variance_1 = np.mean(class1['dif_sqr'])
variance_2 = np.mean(class2['dif_sqr'])

f.write('\nVariance of feature 2 of each class:\n')
f.write('Class 1 variance: '+str(variance_1)+'\n')
f.write('Class 2 variance: '+str(variance_2)+'\n')

# ================================  Discriminant Functions   ====================================
posterior()

labels = [1,2]

conf_matrix = confusion_matrix(test['class'], test['prediction'],labels=labels)

cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
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