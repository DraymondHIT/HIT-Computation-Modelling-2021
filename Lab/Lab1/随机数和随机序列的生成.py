import numpy as np

# N为随机数个数
N = 1000

# [-1, 1]的均匀分布随机数列生成
X = np.random.uniform(-1, 1, N)
Y = np.random.uniform(-1, 1, N)

# 蒙特卡罗投点法
in_circle = 0
for i in range(N):
    if pow(X[i], 2) + pow(Y[i], 2) <= 1:
        in_circle += 1

# 估计的圆周率
print("{:.3f}".format(in_circle / N * 4))
