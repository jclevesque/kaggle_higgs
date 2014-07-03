# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import math

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC

import hyperopt
from hyperopt import fmin, tpe, hp
import hpsklearn


def main(test=False):
    time_start = time.time()

    # Load training data
    data_train = np.loadtxt('training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )

    # Pick a random seed for reproducible results. Choose wisely!
    np.random.seed(42)
    # Random number for training/validation splitting
    r = np.random.rand(data_train.shape[0])
     
    # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
    # First 90% are training
    split = 0.75
    train = r < split
    Y_train = data_train[:,32][train]
    X_train = data_train[:,1:31][train]
    #W_train = data_train[:,31][r<0.9]

    # Last 10% are validation
    val = r >= split
    Y_valid = data_train[:,32][val]
    X_valid = data_train[:,1:31][val]
    W_valid = data_train[:,31][val]
     
    estimator = hpsklearn.HyperoptEstimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        classifier=hpsklearn.components.any_classifier('clf'),
        algo=hyperopt.tpe.suggest,
        trial_timeout=30.0, # seconds
        max_evals=15,
    )

    fit_iterator = estimator.fit_iter(X_train, Y_train)
    fit_iterator.next()
    #plot_helper = hpsklearn.demo_support.PlotHelper(estimator,
    #                                                mintodate_ylim=(-.01, .05))
    while len(estimator.trials.trials) < estimator.max_evals:
        fit_iterator.send(1) # -- try one more model
    #    plot_helper.post_iter()
    #plot_helper.post_loop()

    # -- Model selection was done on a subset of the training data.
    # -- Now that we've picked a model, train on all training data.
    estimator.retrain_best_model_on_full_data(X_train, Y_train)
    print('Best preprocessing pipeline:')
    for pp in estimator._best_preprocs:
        print(pp)
    print()
    print('Best classifier:\n', estimator._best_classif)
    test_predictions = estimator.predict(X_valid)
    acc_in_percent = 100 * np.mean(test_predictions == Y_valid)
    print()
    print('Prediction accuracy in generalization is %.1f%%' % acc_in_percent)

    # Get the probaility output from the trained method, using the 10% for testing
    #prob_predict_train = np.mean(estimator.predict(X_train) == Y_train)
    #prob_predict_valid = np.mean(estimator.predict(X_valid) == Y_valid)

    # Experience shows me that choosing the top 15% as signal gives a good AMS score.
    # This can be optimized though!
    #pcut = np.percentile(prob_predict_train, 0.15)

    # This are the final signal and background predictions
    #Yhat_valid = prob_predict_valid > pcut
    Yhat_valid = estimator.predict(X_valid)

    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
    TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)

    # s and b for the training 
    s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
    b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )

    # Now calculate the AMS scores
    def AMSScore(s,b):
        return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    print('   - AMS based on 10% validation sample:', AMSScore(s_valid, b_valid))

    # Generate predictions.csv
    if test:
        data = pd.read_csv("test.csv")
        X_test = data.values[:, 1:]
        ids = data.EventId
        d = gbc.predict_proba(X_test)[:, 1]
        r = np.argsort(d) + 1
        p = np.empty(len(X_test), dtype=np.object)
        p[d > pcut] = 's'
        p[d <= pcut] = 'b'
        df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
        df.to_csv("predictions.csv", index=False, cols=["EventId", "RankOrder", "Class"])

    return {'loss': -AMSScore(s_valid, b_valid), 'status': hyperopt.STATUS_OK,
        'eval_time': time.time() - time_start}

if __name__ == '__main__':
    best = main()
    print(best)