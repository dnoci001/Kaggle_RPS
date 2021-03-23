# Kaggle_RPS
Kaggle's Rock Paper Scissors Competition

The goal of the competition is to create an agent that will compete against other agents playing Rock Paper Scissors(RPS).

![](https://github.com/dnoci001/Kaggle_RPS/blob/main/images/trophy.jpg)

[This agent finished in 15th place of 1663 competitors](https://www.kaggle.com/c/rock-paper-scissors/leaderboard)


# Model
The ideal model is both predictive of our opponent and opaque to predictions from our opponent.  

Extra Trees was found to give good probability distribution for opponents moves. The predicted probabilities where then used as weights
to sample a predicted move from np.choice, this provided the necessary element of opaqueness to our model. 

# Performance

local testing of agent's performance against common agents. 
WIN: (20,1000)
TIE: [-20,20]
LOSS: (-1000,-20)

![](https://github.com/dnoci001/Kaggle_RPS/blob/main/images/performance_mat.png)

We see that our Extra Trees Agent Wins against the same agents that the Geometry bot does, but has a smaller margin of victory. This is ultimately the
trade-off with keeping our agent as opaque as possible while still being predictive.
