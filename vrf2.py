import sys
import time
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def init_forest():
    global forest
    forest = RandomForestClassifier()

def warmup_strategy():
    action = int(np.random.randint(3))
    return int(action)

def init_train(step):
    global data,train,nfeatures
    train = np.array(data[:nfeatures]).reshape(1,-1)
    for i in range(1,step-nfeatures):
        train = np.concatenate((train,data[i:i+nfeatures].reshape(1,-1)))

def build_train(step):
    global data,train,test,moves,nfeatures
    i = step-nfeatures - 1
    train = np.concatenate((train,data[i:i+nfeatures].reshape(1,-1)))
         
    moves = np.array(data[nfeatures:step])
    test = np.array(data[-1*nfeatures:]).reshape(1,-1)
       
def predict(step):
    global data,train,test,moves,forest
    max_inc = 300
    max_dinc = 320
    
    ## Collect start time 
    start = int(round(time.time() * 1000))
    
    ## Fit Random Forest and get probablities in dict form
    forest.fit(train,moves)  
    proba = dict(zip(forest.classes_.astype(int), forest.predict_proba(test)[0]))
    
    ## set probablity for missing classes = 0 and squash low probability guesses
    for c in range(3):
        if not c in proba:
            proba[c] = 0.0 
        else:
            proba[c] = proba[c]-0.2 

    ## populate an array of votes from which we will randomly choose from
    votes = np.empty(0)
    for a,p in proba.items(): 
        if p > 0.0:
            for i in range(int(100*p)):
                votes = np.append(votes,int(a))

    ## if there is more than one vote randomly choose from it else just choose random action
    if len(votes) > 0:           
        action = int(votes[np.random.randint(len(votes)-1)])  
    else:
        action = warmup_strategy()
    
    ## Collect end time and increase or decrease number of trees
    end = int(round(time.time() * 1000))
    pred_time = end-start
    if pred_time < max_inc:
        forest.n_estimators += 5
    elif pred_time > max_dinc:
        forest.n_estimators -= 5

    return action

def agent(observation, configuration):
    global data,votestrain,nvotes,nfeatures,forest,guess,correct
    ndiscard = 20 ## assume that the first ndiscard actions are garbage/random actions that might polute our model
    nwarmup = 50 ## period after which we will begin predicting actions
    nfeatures = 7 ## window used in out model
    step = observation.step
    
    if step == 0:
        data = np.empty(0)
        votestrain = np.empty(0)
        init_forest()
    else:
        if step >= ndiscard:
            lastmove = observation.lastOpponentAction
            data = np.append(data,[lastmove])   
            
    if step < nwarmup:
        action = warmup_strategy()
        if (step+1) == nwarmup:
            init_train(step-ndiscard)
    else:
        build_train(step-ndiscard)       
        action = predict(step)

    action = int((action + 1) % 3)

    return action
