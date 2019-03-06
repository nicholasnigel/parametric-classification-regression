from classifier import Classifier

cl = Classifier('input_1.csv')

f = open('prob1.txt','w')
f.write("Prior Probability:\n"+str(cl.prior_prob)+"\n")
