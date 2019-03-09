from classifier import Classifier




cl = Classifier('input_1.csv')
cl.split_data(0.8)
priorprob = cl.calculate_prior()
estimate_p = cl.bernoulli_estimator('feature_value')


print(cl.bernoulli_likelihood(estimate_p, 0))
