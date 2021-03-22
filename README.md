# Kaggle_RPS
Kaggle's Rock Papper Scissors Competition

The goal of the competition is to create an agent that will compete against other agents playing Rock Paper Scissors.

Each round we will have a few hundred milliseconds to predict our opponent's next move. Due to time constraints we will be exploring classical ML models.
Ideally we will have a model that is predictive of our opponent and opaque to predictions from our opponent.
A nice middle ground was found to be an implimentation of Extra Trees where the opponenets predicted move was then sampleed from np.choice
weighted by the predicted probabilites of the Extra Trees model.
