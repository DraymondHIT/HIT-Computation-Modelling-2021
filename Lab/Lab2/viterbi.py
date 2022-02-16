import numpy as np
from hmmlearn import hmm

states = ["Sunny", "Rainy"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])

emission_probability = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.4, 0.5]
])

model = hmm.MultinomialHMM(n_components=n_states, n_iter=20)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

_observation_num = [2, 1, 3, 2, 3, 2, 2, 3, 3, 1, 2, 1, 1, 1, 2, 3, 3, 3, 3, 2]
seen = np.array([[i-1 for i in _observation_num]]).T
_, hidden = model.decode(seen, algorithm="viterbi")
print("The hidden states", hidden)
