import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

NumDataPerClass = 200

m1 = [[0, 5]]
m2 = [[5, 0]]
C = [[2, 1], [1, 2]]

A = np.linalg.cholesky(C)

U1 = np.random.randn(NumDataPerClass, 2)
X1 = U1 @ A.T + m1

U2 = np.random.randn(NumDataPerClass, 2)
X2 = U2 @ A.T + m2

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X1[:,0], X1[:,1], c="c", s=4)
ax.scatter(X2[:,0], X2[:,1], c="m", s=4)

# 拼接两个类别中的数据坐标
X = np.concatenate((X1, X2), axis = 0)

# 创建标签
labelPos = np.ones(NumDataPerClass)
labelNeg = -1.0 * np.ones(NumDataPerClass)
y = np.concatenate((labelPos, labelNeg))

# 将数据分割成训练集和测试集

# 对数据和标签进行随机排序
rIndex = np.random.permutation(2 * NumDataPerClass)
Xr = X[rIndex,]
yr = y[rIndex]

# 一半做训练 一半做测试
X_train = Xr[0 : NumDataPerClass]
y_train = yr[0 : NumDataPerClass]
X_test = Xr[NumDataPerClass : 2 * NumDataPerClass]
y_test = yr[NumDataPerClass : 2 * NumDataPerClass]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

Ntrain = NumDataPerClass
Ntest = NumDataPerClass


# 计算基于weights参数的模型的正确率
def PercentCorrect(Inputs, targets, weights):
    N = len(targets)
    nCorrect = 0

    for n in range(N):
        OneInput = Inputs[n, :]
        if (targets[n] * np.dot(OneInput, weights) > 0):
            nCorrect += 1

    return 100 * nCorrect / N

# 随机初始化weights
w = np.random.randn(2)
print(w)

# 计算初始模型的正确率
print('Initial Percentage Correct: %6.2f' %(PercentCorrect(X_train, y_train, w)))

MaxIter = 1000  # 迭代次数
alpha = 0.01  # 学习率
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

# 开始迭代
for iter in range(MaxIter):
    # 随机选择一个训练数据
    r = np.floor(np.random.rand() * Ntrain).astype(int)
    x = X_train[r, :]

    # 如果模型结果错误，将分割线靠近该训练点
    if y_train[r] * np.dot(x, w) < 0:
        w += alpha * y_train[r] * x

    # 分别计算该模型在训练集和测试集中的正确率
    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);

    # if P_train[iter] == 100 and P_test[iter] == 100:
    #     break

print('Percentage Correct After Training: %6.2f  %6.2f' % (
PercentCorrect(X_train, y_train, w), PercentCorrect(X_test, y_test, w)))
print(w)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(range(MaxIter), P_train, 'b', label = "Training")
ax.plot(range(MaxIter), P_test, 'r', label = "Test")
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning')
ax.set_ylabel('Training and Test Accuracies', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)
plt.savefig('learningCurves.eps')

model = Perceptron()
model.fit(X_train, y_train)
yh_train = model.predict(X_train)
print("Accuracy on training set: %6.2f" %(accuracy_score(yh_train, y_train)))

yh_test = model.predict(X_test)
print("Accuracy on test set: %6.2f" %(accuracy_score(yh_test, y_test)))

if accuracy_score(yh_test, y_test) > 0.99:
    print("Wow, Perfect Classification on Separable dataset!")
