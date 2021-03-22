import sys
import time
import random
import numpy as np
import collections
from numpy.random import choice
from sklearn.tree import DecisionTreeClassifier

def init_forest():
    global forest,nest,k,min_samples,discard,primelist,dp
    k = 8
    mf = 1
    min_samples = 300
    
    
    discard = 101
    forest = list()
    for i in range(2000):
        forest.append(DecisionTreeClassifier(splitter='random',max_features=mf,random_state=np.random.randint(1000000)))


def construct_local_features(rollouts):
    features = np.array(rollouts['actions'])
    features = np.append(features, rollouts['opp-actions'])
    return features

def construct_global_features(rollouts):
    features = []
    for key in ['actions', 'opp-actions']:
        for i in range(3):
            actions_count = np.mean([r == i for r in rollouts[key]])
            features.append(actions_count)
    
    return np.array(features)

def construct_features(short_stat_rollouts, long_stat_rollouts):
    lf = construct_local_features(short_stat_rollouts)
    gf = construct_global_features(long_stat_rollouts)
    features = np.concatenate([lf, gf])
    return features

def predict_opponent_move(train_data, test_sample):
    global forest,dp,start,step,min_samples
    max_inc = 950
    tproba = dict()
    tproba[0] = 0.0
    tproba[1] = 0.0
    tproba[2] = 0.0

    for tree in forest:
        tree.fit(train_data['x'], train_data['y'])     
        proba = dict(zip(tree.classes_.astype(int), tree.predict_proba(test_sample)[0]))
        for c in proba:
            tproba[c] = tproba[c] + proba[c]
        end = int(round(time.time() * 1000))
        if end - start > max_inc:
            break
            
    votes = np.array([tproba[0],tproba[1],tproba[2]])
    vsum = np.sum(votes)
    votes /= vsum
    
    action = choice([0,1,2],p=votes)

    return action

def update_rollouts_hist(rollouts_hist, last_move, opp_last_action):
    rollouts_hist['steps'].append(last_move['step'])
    rollouts_hist['actions'].append(last_move['action'])
    rollouts_hist['opp-actions'].append(opp_last_action)
    return rollouts_hist

def warmup_strategy(observation, configuration):
    global rollouts_hist, last_move,discard

    action = int(np.random.randint(3))
    if observation.step == 0:
        last_move = {'step': 0, 'action': action}
        rollouts_hist = {'steps': [], 'actions': [], 'opp-actions': []}
    elif observation.step > discard:
        rollouts_hist = update_rollouts_hist(rollouts_hist, last_move, observation.lastOpponentAction)
        last_move = {'step': observation.step, 'action': action}
    return int(action)

def init_training_data(rollouts_hist, k):
    for i in range(len(rollouts_hist['steps']) - k + 1):
        short_stat_rollouts = {key: rollouts_hist[key][i:i+k] for key in rollouts_hist}
        long_stat_rollouts = {key: rollouts_hist[key][:i+k] for key in rollouts_hist}
        features = construct_features(short_stat_rollouts, long_stat_rollouts)        
        data['x'].append(features)
    test_sample = data['x'][-1].reshape(1, -1)
    data['x'] = data['x'][:-1]
    data['y'] = rollouts_hist['opp-actions'][k:]
    return data, test_sample

def agent(observation, configuration):
    global rollouts_hist, last_move, data, test_sample, nest, k, min_samples, discard, start,step
    step = observation.step
    start = int(round(time.time() * 1000))
    if observation.step == 0:
        data = {'x': [], 'y': []}
        init_forest()
    # if not enough data -> randomize
    if observation.step <= discard + min_samples + k:
        return warmup_strategy(observation, configuration)
    # update statistics
    rollouts_hist = update_rollouts_hist(rollouts_hist, last_move, observation.lastOpponentAction)
    # update training data
    if len(data['x']) == 0:
        data, test_sample = init_training_data(rollouts_hist, k)
    else:        
        short_stat_rollouts = {key: rollouts_hist[key][-k:] for key in rollouts_hist}
        features = construct_features(short_stat_rollouts, rollouts_hist)
        data['x'].append(test_sample[0])
        data['y'] = rollouts_hist['opp-actions'][k:]
        test_sample = features.reshape(1, -1)
        
    # predict opponents move and choose an action
    next_opp_action_pred = predict_opponent_move(data, test_sample)
    action = int((next_opp_action_pred + 1) % 3)
    last_move = {'step': observation.step, 'action': action}
    return action
