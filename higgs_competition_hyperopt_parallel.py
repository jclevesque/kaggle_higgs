# -*- coding: utf-8 -*-

import time
import math

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
import pandas as pd

import hyperopt
from hyperopt import fmin, tpe, hp

def eval(params, test=False):
    time_start = time.time()

    # Load training data
    #print('Loading training data.')
    data_train = np.loadtxt('debug.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
     
    # Pick a random seed for reproducible results. Choose wisely!
    np.random.seed(42)
    # Random number for training/validation splitting
    r = np.random.rand(data_train.shape[0])
     
    # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
    #print('Assigning data to numpy arrays.')
    # First 90% are training
    Y_train = data_train[:,32][r<0.9]
    X_train = data_train[:,1:31][r<0.9]
    #W_train = data_train[:,31][r<0.9]
    # Last 10% are validation
    Y_valid = data_train[:,32][r>=0.9]
    X_valid = data_train[:,1:31][r>=0.9]
    W_valid = data_train[:,31][r>=0.9]
     
    # Train the GradientBoostingClassifier using our good features
    #print('Training classifier (this may take some time!)')
    gbc = GBC(n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=int(params['max_features']),
        learn_rate=params['learn_rate'])
    gbc.fit(X_train,Y_train) 
     
    # Get the probaility output from the trained method, using the 10% for testing
    prob_predict_train = gbc.predict_proba(X_train)[:,1]
    prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
     
    # Experience shows me that choosing the top 15% as signal gives a good AMS score.
    # This can be optimized though!
    pcut = np.percentile(prob_predict_train, params['pcut'])
     
    # This are the final signal and background predictions
    #Yhat_train = prob_predict_train > pcut 
    Yhat_valid = prob_predict_valid > pcut
     
    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    #TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.9)
    #TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.9)
    TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
    TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)
     
    # s and b for the training 
    #s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
    #b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
    s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
    b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
     
    # Now calculate the AMS scores
    #print('Calculating AMS score for a probability cutoff pcut=',pcut)
    def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    #print('   - AMS based on 90% training   sample:',AMSScore(s_train,b_train))
    print('   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid))

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

    return {'loss': -AMSScore(s_valid, b_valid), 'status':hyperopt.STATUS_OK,
        'eval_time': time.time() - time_start}

my_space = {
    'n_estimators': hp.quniform('n_estimators', 1, 200, 1), 
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10000, 1),
    'max_features': hp.quniform('max_features', 1, 30, 1),
    'learn_rate': hp.loguniform('learn_rate', -2, 2),
    'pcut': hp.uniform('pcut', 50, 100)
    }

trials = MongoTrials('mongo://localhost:1234/kaggle_db/jobs', exp_key='GBC_debug')
best = fmin(fn=eval,
    space=my_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
print best