# Kaggle_RPS
Kaggle's Rock Papper Scissors Competition

The goal of the competition is to create an agent that will compete against other agents playing Rock Paper Scissors(RPS).


# Model
The ideal model is both predictive of our opponent and opaque to predictions from our opponent.  

Extra Trees was found to give good probabily distribution for opponents moves. The predicted probabilites where then used as weights
to sample a predicted move from np.choice, this provided the nessecary element of opaqueness to our model. 

# Performance
