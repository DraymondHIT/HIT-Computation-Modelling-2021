import numpy as np
import math
import matplotlib.pyplot as plt

# 样本点个数
N = 1000

X = []
num = [i for i in range(1, N+1)]
mean = []
var = []

for i in range(N):
    # 正态分布随机数生成
    x = np.random.normal(10, math.sqrt(5))
    X.append(x)

    # 计算均值与方差
    mean.append(np.mean(X))
    var.append(np.var(X))


# 作图
plt.subplot(1, 2, 1)
plt.plot(num, mean)
plt.title('mean')

plt.subplot(1, 2, 2)
plt.plot(num, var)
plt.title('var')
plt.show()


# 模拟坦克到达的数量
for i in range(3):
    print(f'minute {i+1}: {np.random.poisson(4)}')


# 模拟坦克到达的时刻
time = 0
time += np.random.exponential(0.25)
num = 1
print(f'tank {num}: {time}')
while time < 3:
    time += np.random.exponential(0.25)
    num += 1
    if time < 3:
        print(f'tank {num}: {time}')

