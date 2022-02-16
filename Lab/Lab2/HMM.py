import numpy as np
import bisect


class HMM:
    def __init__(self, _states, _observations, _transition_probability, _emission_probability, _initial_state):
        self.states = _states
        self.observations = _observations
        self.transition_probability = _transition_probability
        self.emission_probability = _emission_probability
        self.initial_state = _initial_state

    @staticmethod
    def get_random_state(distribution):
        temp = distribution[0]
        for i in range(1, len(distribution)):
            temp += distribution[i]
            distribution[i] = temp

        X = np.random.uniform(0, 1)
        return bisect.bisect(distribution, X)

    def hidden_predict(self, n_iter):
        hidden_state = self.initial_state.T
        emission_states = []
        for i in range(n_iter):
            emission_state = self.get_random_state(hidden_state.squeeze().tolist())
            emission_states.append(emission_state)
            hidden_state = self.transition_probability[emission_state]

        return [self.states[i] for i in emission_states]

    def emission_predict(self, hidden_seq):
        n_iter = len(hidden_seq)
        hidden_seq = [states.index(i) for i in hidden_seq]
        emission_states = []
        for i in range(n_iter):
            emission_state = self.get_random_state(emission_probability[hidden_seq[i]].squeeze().tolist())
            emission_states.append(emission_state)

        return [self.observations[i] for i in emission_states]

    def percentage(self, observation_seq, hidden_seq):
        assert len(observation_seq) == len(hidden_seq)
        n_iter = len(observation_seq)
        hidden_seq = [states.index(i) for i in hidden_seq]
        observation_seq = [observations.index(i) for i in observation_seq]

        # forward
        alpha = self.initial_state * self.emission_probability[:, observation_seq[0]].reshape(-1, 1)
        for i in range(1, n_iter):
            alpha = self.transition_probability.T.dot(alpha) * self.emission_probability[:, observation_seq[i]].reshape(-1, 1)
        p_observation = np.sum(alpha)

        p_hidden = self.initial_state.squeeze()[hidden_seq[0]]
        for i in range(1, n_iter):
            p_hidden *= self.transition_probability[hidden_seq[i - 1], hidden_seq[i]]

        p_condition = 1
        for i in range(n_iter):
            p_condition *= self.emission_probability[hidden_seq[i], observation_seq[i]]

        return p_hidden * p_condition / p_observation

    def viterbi(self, observation_seq):
        n_iter = len(observation_seq)
        observation_seq = [observations.index(i) for i in observation_seq]

        delta = np.zeros((n_iter, self.initial_state.shape[0]))
        phi = np.zeros((n_iter, self.initial_state.shape[0]))
        delta[0] = self.initial_state.T * self.emission_probability.T[observation_seq[0]]

        for i in range(1, n_iter):
            temp = delta[i-1] * self.transition_probability.T
            _index = np.argmax(temp, axis=1)
            phi[i] = np.array(_index)
            new = np.zeros((1, len(_index)))
            for j in range(len(_index)):
                new[0, j] = temp[j, _index[j]]
            delta[i] = new * self.emission_probability.T[observation_seq[i]]

        ans_seq = [np.argmax(delta[-1])]
        for i in range(n_iter-1, 0, -1):
            ans_seq.append(int(phi[i, ans_seq[-1]]))
        ans_seq = list(reversed(ans_seq))

        return [self.states[i] for i in ans_seq]


states = ["Sunny", "Rainy"]
observations = ["walk", "shop", "clean"]

start_probability = np.array([
    [0.6],
    [0.4]
])
transition_probability = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])
emission_probability = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.4, 0.5]
])

model = HMM(states, observations, transition_probability, emission_probability, start_probability)

# 随机生成天气序列
weather_seq = model.hidden_predict(20)
print("随机生成的天气序列：")
print(weather_seq)
print([states.index(weather)+1 for weather in weather_seq])

# 根据天气序列生成活动序列
activity_seq = model.emission_predict(weather_seq)
print("根据上述天气序列生成的活动序列：")
print(activity_seq)
print([observations.index(activity)+1 for activity in activity_seq])

# 已知观测序列 推算天气序列概率
percentage = model.percentage(["walk", "shop", "clean"], ["Sunny", "Rainy", "Rainy"])
print("Percentage: ", percentage)

# 已知观测序列 推算出现概率最大的天气概率
_observation_num = [2, 1, 3, 2, 3, 2, 2, 3, 3, 1, 2, 1, 1, 1, 2, 3, 3, 3, 3, 2]
_observation_seq = [observations[i-1] for i in _observation_num]
most_probable_weather_seq = model.viterbi(_observation_seq)
print("概率最大的天气序列：")
print(most_probable_weather_seq)
print([states.index(weather)+1 for weather in most_probable_weather_seq])


# states = ["1", "2", "3"]
# observations = ["hong", "bai"]
#
# A = np.array([
#     [0.5, 0.2, 0.3],
#     [0.3, 0.5, 0.2],
#     [0.2, 0.3, 0.5]
# ])
#
# B = np.array([
#     [0.5, 0.5],
#     [0.4, 0.6],
#     [0.7, 0.3]
# ])
#
# P = np.array([
#     [0.2],
#     [0.4],
#     [0.4]
# ])
#
# _model = HMM(states, observations, A, B, P)
# hidden_seq = _model.viterbi(["hong", "bai", "hong"])
# print(hidden_seq)

