import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Classifier:
    def __init__(self,filename):
        #       reading in the data
        self.input = pd.read_csv(filename)
        #       splitting data into training and testing

        self.train , self.test = train_test_split(self.input, test_size=0.2, shuffle=False)
        #       Prior
        self.prior = self.train['class'].value_counts()
        #       Prior Probability
        self.prior_prob = self.prior/len(self.train)
    
    def bernoulli_estimator(self,feature):
        # find all 
