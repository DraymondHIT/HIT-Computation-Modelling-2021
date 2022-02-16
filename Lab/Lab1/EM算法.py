import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Gaussian(x, mu, sigma):
    return 1 / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5)) * np.exp(
        -0.5 * (x - mu).dot(np.linalg.pinv(sigma)).dot((x - mu).T))


def compute_Gamma(X, mu, sigma, alpha):
    n_samples = X.shape[0]
    n_clusters = len(alpha)
    gamma = np.zeros((n_samples, n_clusters))
    p = np.zeros(n_clusters)
    pm = np.zeros(n_clusters)
    for i in range(n_samples):
        for j in range(n_clusters):
            p[j] = Gaussian(X[i], mu[j], sigma[j])
            pm[j] = alpha[j] * p[j]
        for j in range(n_clusters):
            gamma[i, j] = pm[j] / np.sum(pm)
    return gamma


class GMM:
    # 初始化
    def __init__(self, n_clusters, iter=50):
        self.n_clusters = n_clusters
        self.iter = iter
        self.mu = 0
        self.sigma = 0
        self.alpha = 0

    # EM算法
    def fit(self, data):
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # 初始化alpha, mu, sigma
        alpha = np.ones(self.n_clusters) / self.n_clusters
        mu = np.array([[.403, .237], [.714, .346], [.532, .472]])
        sigma = np.full((self.n_clusters, n_features, n_features), np.diag(np.full(n_features, 0.1)))

        for i in range(self.iter):
            gamma = compute_Gamma(data, mu, sigma, alpha)
            for j in range(self.n_clusters):
                # 计算新均值向量
                mu[j] = np.sum(data * gamma[:, j].reshape((n_samples, 1)), axis=0)\
                        / np.sum(gamma, axis=0)[j]

                # 计算新协方差矩阵
                sigma[j] = 0
                for k in range(n_samples):
                    sigma[j] += (data[k].reshape((1, n_features)) - mu[j]).T.dot(
                        (data[k] - mu[j]).reshape((1, n_features))) * gamma[k, j]
                sigma[j] = sigma[j] / np.sum(gamma, axis=0)[j]

            # 计算新混合系数
            alpha = np.sum(gamma, axis=0) / n_samples

        # 保存参数
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    # 预测
    def predict(self, data):
        # 计算簇标记
        lambd = compute_Gamma(data, self.mu, self.sigma, self.alpha)

        # 分类
        cluster_results = np.argmax(lambd, axis=1)

        print('alpha:')
        print(self.alpha)
        print('mu:')
        print(self.mu)
        print('sigma:')
        print(self.sigma)

        return cluster_results


data = pd.read_excel('data.xlsx', header=None)
data = data.values

model = GMM(3)
model.fit(data)
result = model.predict(data)
plt.scatter(data[:, 0], data[:, 1], c=result)
plt.scatter(model.mu[:, 0], model.mu[:, 1], marker='+', color='red')
plt.show()
