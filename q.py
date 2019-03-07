from classifier import Classifier

cl = Classifier('input_1.csv')

f = open('prob1.txt','w')
f.write("Prior Probability:\n"+str(cl.prior_prob)+"\n")
p = cl.bernoulli_estimator('feature_value')
likelihood = cl.bernoulli_likelihood(p,1)


