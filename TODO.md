# minibatch for training
# parallel restart
# entirely initialize experiment in GPExp (likelihood, mean, covar)
# support loading GPExp from ini
# support dumping GPExp to ini
# snapshotting
# provide one implementation of EarlyStoppingCriterion that only uses training data (without validation)
# support dumping of TrainStats to file

# every esc should keep track of the current_best_model
# this motherfucker should be returned on every evaluate() call
# (together with the best_epoch).
# and saved when early_stopping_criterion.is_verified
