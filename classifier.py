import pandas as pd 
import numpy as np
from math import pow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Classifier:
    def __init__(self,filename):
        """"
        Class initializer
        """
        #       reading in the data
        
        self.input = pd.read_csv(filename)
        #       splitting data into training and testing

        self.train , self.test = train_test_split(self.input, test_size=0.2, shuffle=False)
        #       Prior
        self.prior = self.train['class'].value_counts()
        self.unique_class = self.train['class'].unique()
        #       Prior Probability
        self.prior_prob = self.prior/len(self.train)
        
        self.classes = {i:0 for i in self.train['class'].unique()}
        for c in self.train['class']:
            self.classes[c] = self.train[  (self.train['class'] == c) ]


    def bernoulli_estimator(self,feature):
        """
        Predicts the estimate probability p when x=1 for each class of every 'feature' 

        Parameters:
            param str feature: the name of column you wish to estimate from 
        Returns:
            pr: An estimate of p from this column for each class
        """
        pr = {i:0 for i in self.train['class'].unique()}
        for i in pr.keys():
            pr[i] = len(self.train[ (self.train['class'] == i) & (self.train[feature] == 1) ]) / self.prior[i]
        return pr

    def bernoulli_likelihood(self,p,x):
        """
        Calculates the likelihood of a certain class: p(x|Ci)
        Parameters
        ----------
        p: dict
            Dictionary of each estimation of p for each class

        Returns
        -------
        g: dict
            Dictionary containing a discriminant function
            
        """
        g = {i:0 for i in self.train['class'].unique()}
        for i in g.keys():
            g[i] = pow( p[i],x)* pow(1-p[i], 1-x)

        return g
        
    def discriminant_function(self, likelihood):
        """
        Calculate the overall discriminant function by multiplying likelihood to prior of each class
        
        Parameters
        ----------
        likelihood: dictionary 
            The likelihoods for each class

        Returns
        -------
        key: index
            The key of largest value in dictionary
        """    
        g = {i:0 for i in self.unique_class}
        for i in self.unique_class:
            g[i] = likelihood[i] * self.prior_prob[i]
        return(max(g, key= g.get))



        
        
