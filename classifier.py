import pandas as pd 
import numpy as np
from math import pow

class Classifier:
    """Classifier CLass"""
    def __init__(self,filename):
        """
        class initializer

        >>> Classifier('input_1.csv')
        Parameter
        ---------
            - filename: the file you're trying to input
        """
        self.data = pd.read_csv(filename)

        
    def split_data(self,training_size):
        """
        splitting the input data into training and testing data, normalized to 1
    
        Parameter
        ---------
            > training_size: number out of 1 where how much of the data should be put as the training data
        """
        num_data = int(training_size * len(self.data))
        num_test = int(1-training_size * len(self.data)) -1
        self.train = self.data.head(num_data)
        self.test = self.data.tail(num_test)

        return self.train,self.test
    
    def calculate_prior(self):
        """
        calculating prior probability of each class on training data

        Returns
        -------
            > dictionary of prior probability of each class
        
        """

        self.priors = self.train['class'].value_counts()
        self.prior_prob = self.priors/len(self.train)

        return self.prior_prob
    
    def bernoulli_estimator(self, feature):
        """
        Estimates the likelihood of x=1 given for each class 

        Returns
        -------
            > p: dict of probability estimation that estimates Class i = 1, x =1

        """
        p = {i:0 for i in self.train['class'].unique()}
        #       Estimate p
        for i in p.keys():
            p[i] = len(self.train[ (self.train['class']== i) & (self.train[feature]==1)]) / self.priors[i]

        return p 

    
    def bernoulli_likelihood(self,p,x ):
        """
        Calculates the likelihood of getting each class with x = 1
        
        Parameters
        ----------
            > p: dictionary of probability (an estimate)
            
        """

        pXC = {i:0 for i in self.train['class'].unique()}
        for i in pXC.keys():
            pXC[i] = pow(p[i], x) * pow(1-p[i], 1-x)
        return pXC
    
    def prediction(self):
        """
        Predicting 
        """


        

