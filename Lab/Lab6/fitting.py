import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 数据点个数
n_samples = 20

# 外点个数
n_outliers = 0

# 多项式阶数
n_orders = 1

# 选项
# 0: 直线
# 1: 正弦曲线
choice = 0

# 标准曲线
x0 = np.linspace(0, 1, 100)
y0 = np.sin(2 * np.pi * x0) if choice else x0 + 1
y1 = x0 - 1
y2 = x0 + 3


# 采样函数
def generate_data(N, option):
    x = np.linspace(0, 1, N)
    if option:
        y = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.2, size=N)  # 增加高斯噪声
    else:
        y = x + 1 + np.random.normal(loc=0, scale=0.2, size=N)
    return x.reshape(N, 1), y.reshape(N, 1)


def generate_parallel_data(N):
    x = np.linspace(0, 1, N)
    _x = np.concatenate((x, x))
    y = x - 1 + np.random.normal(loc=0, scale=0.1, size=N)
    y = np.concatenate((y, x + 1 + np.random.normal(loc=0, scale=0.1, size=N)))
    y = np.concatenate((y, x + 3 + np.random.normal(loc=0, scale=0.1, size=N)))
    x = np.concatenate((_x, x))
    return x.reshape(-1, 1), y.reshape(-1, 1)


def add_outliers(N, option):
    x = np.random.uniform(1, 1.5, N)
    y = np.ones((N, 1))
    return x.reshape(N, 1), y


# 最小二乘法
def OLS_fitting(M, x, y, lambd=0):
    # 计算X
    order = np.arange(M + 1).reshape((1, -1))
    X = x ** order

    # 计算W
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) +
                                    lambd * np.identity(M + 1)), X.T), y)
    # loss = np.sum(0.5 * np.dot((y - np.dot(X, W)).T, y - np.dot(X, W)) + 0.5 * lamda * np.dot(W.T, W))
    X0 = x0.reshape(-1, 1) ** order
    return np.dot(X0, W), W


# RANSAC
def RANSAC_fitting(M, x, y, sigma=0.25):
    # 初始化参数
    n_iters = 100
    mini_batch = 10
    n_inliers_max = 0
    best_W = np.zeros((M + 1, 1))
    n_samples = x.shape[0]
    P = 0.99
    order = np.arange(M + 1).reshape((1, -1))

    # 估计迭代次数
    for i in range(mini_batch):
        # 求解曲线表达式
        _indexs = np.random.randint(0, n_samples, M + 1)
        A = x[_indexs, 0].reshape(-1, 1) ** order
        b = y[_indexs, 0].reshape(-1, 1)
        try:
            W = np.linalg.solve(A, b)
        except:
            continue

        # 统计内点数目
        n_inliers = 0
        for j in range(n_samples):
            estimate_y = (x[j, 0].reshape(-1, 1) ** order).dot(W)
            if np.abs(estimate_y - y[j, 0]) <= sigma:
                n_inliers += 1

        # 判断当前的模型是否比之前估算的模型好
        if n_inliers > n_inliers_max:
            # 估计迭代次数
            n_iters = int(np.log(1 - P) / np.log(1 - pow(n_inliers / n_samples, M + 1)))
            n_inliers_max = n_inliers
            best_W = W

    # 开始迭代
    for i in range(n_iters):
        # 求解曲线表达式
        _indexs = np.random.randint(0, n_samples, M + 1)
        A = x[_indexs, 0].reshape(-1, 1) ** order
        b = y[_indexs, 0].reshape(-1, 1)
        try:
            W = np.linalg.solve(A, b)
        except:
            continue

        # 统计内点数目
        n_inliers = 0
        for j in range(n_samples):
            estimate_y = (x[j, 0].reshape(-1, 1) ** order).dot(W)
            if np.abs(estimate_y - y[j, 0]) <= sigma:
                n_inliers += 1

        # 判断当前的模型是否比之前估算的模型好
        if n_inliers > n_inliers_max:
            n_inliers_max = n_inliers
            best_W = W

    X0 = x0.reshape(-1, 1) ** order
    return np.dot(X0, best_W), best_W


def normal():
    # 获取带噪声的数据
    train_x, train_y = generate_data(n_samples, option=choice)

    # 生成外点
    outlier_x, outlier_y = add_outliers(n_outliers, option=choice)

    train_x = np.row_stack((train_x, outlier_x))
    train_y = np.row_stack((train_y, outlier_y))

    # 四种方式获取y和W
    y_1, W1 = OLS_fitting(n_orders, train_x, train_y)
    y_2, W2 = OLS_fitting(n_orders, train_x, train_y, lambd=1e-4)
    y_3, W3 = RANSAC_fitting(n_orders, train_x, train_y, sigma=0.2)

    print(W1)
    print(W2)
    print(W3)

    # 作图
    plt.figure(1, figsize=(8, 5))
    plt.plot(x0, y_1, 'orange', linewidth=2, label='OLS_without_penalty')
    plt.plot(x0, y_2, 'r', linewidth=2, label='OLS_with_penalty')
    plt.plot(x0, y_3, 'pink', linewidth=2, label='RANSAC')
    plt.plot(x0, y0, 'b', linewidth=2, label='base')
    plt.scatter(train_x, train_y, marker='o', edgecolors='b', s=100, linewidth=3)
    plt.title(f'M = {n_orders}, traing_num = {n_samples}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best", fontsize=10)
    plt.show()


def parallel():
    # 生成平行数据
    train_x, train_y = generate_parallel_data(n_samples)
    dataset = np.column_stack((train_x, train_y))

    # 聚类
    db = DBSCAN(eps=1., min_samples=10).fit(dataset)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    OLS_lines = []
    RANSAC_lines = []
    for i in range(n_clusters_):
        _indexs = [index for index in range(len(labels)) if labels[index] == i]
        print(_indexs)
        y, _ = OLS_fitting(n_orders, dataset[_indexs, 0].reshape(-1, 1), dataset[_indexs, 1].reshape(-1, 1))
        OLS_lines.append(y)
        y, _ = RANSAC_fitting(n_orders, dataset[_indexs, 0].reshape(-1, 1), dataset[_indexs, 1].reshape(-1, 1))
        RANSAC_lines.append(y)

    # 作图
    plt.figure(1, figsize=(8, 5))
    plt.plot(x0, y0, 'b', linewidth=2, label='base')
    plt.plot(x0, y1, 'b', linewidth=2)
    plt.plot(x0, y2, 'b', linewidth=2)
    first = True
    for y in OLS_lines:
        if first:
            plt.plot(x0, y, 'r', linewidth=2, label='OLS')
            first = False
        else:
            plt.plot(x0, y, 'r', linewidth=2)
    first = True
    for y in RANSAC_lines:
        if first:
            plt.plot(x0, y, 'pink', linewidth=2, label='RANSAC')
            first = False
        else:
            plt.plot(x0, y, 'pink', linewidth=2)

    plt.scatter(train_x, train_y, marker='o', c=labels+np.ones(3*n_samples, dtype=np.uint8))
    plt.title(f'M = {n_orders}, traing_num = {n_samples}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best", fontsize=10)
    plt.show()


# normal()
parallel()
