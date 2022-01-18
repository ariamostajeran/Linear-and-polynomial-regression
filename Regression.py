import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("GOOGL.csv")

# matrix = []
answers = df.head(len(df.Open) - 10).Open.to_list()

matrix = np.zeros(2)
matrix[0] = 1
matrix[1] = 1
for i in range(1, len(df) - 10):
    rw = np.zeros(2)
    rw[0] = 1
    rw[1] = i
    matrix = np.vstack((matrix, rw))


answers = np.array(answers)
# matrix = np.array(matrix)

# print(matrix.T)
a = np.dot(matrix.transpose(), matrix)
b = np.dot(matrix.transpose(), answers)
x = np.linalg.solve(a, b)

error_sum = 0
print("LinearRegression")
for i in range(len(df) - 10, len(df)):
    print("actual value : ", df.Open[i])
    print("calculated value : ", np.dot([1, i], x))
    # error_sum += abs(df.Open[i] - np.dot([1, i], x))
    print(abs(df.Open[i] - np.dot([1, i], x)))


matrix_2 = np.zeros(3)
matrix_2[0] = 1
matrix_2[1] = 1
matrix_2[2] = 1
for i in range(1, len(df) - 10):
    rw = np.zeros(3)
    rw[0] = 1
    rw[1] = i
    rw[2] = i*i
    matrix_2 = np.vstack((matrix_2, rw))
matrix_2 = np.array(matrix_2)
a1 = np.transpose(matrix_2).dot(matrix_2)
b1 = np.transpose(matrix_2).dot(answers)

x1 = np.linalg.solve(a1, b1)
print(x1)
error_sum1 = 0
predictions_2 = []
print()
print("PolynomalRegression")
for i in range(len(df) - 10, len(df)):
    print("actual value : ", df.Open[i])
    print("calculated value : ", np.dot([1, i, i*i], x1))
    predictions_2.append(np.dot([1, i, i*i], x1))
    error_sum1 += abs(df.Open[i] - np.dot([1, i, i*i], x1))
    print(abs(df.Open[i] - np.dot([1, i, i*i], x1)))

x_cords = range(len(df))
y_cords = []
for i in x_cords:
    y_cords.append(np.dot([1, i], x))
plt.scatter(x_cords, df.Open.to_list(), color='red', label='Open')

plt.scatter(x_cords, y_cords, color='Yellow', label='LinearPrediction')
y_cords1 = []
for i in x_cords:
    y_cords1.append(i*i*x1[2] + i*x1[1] + x1[0])
    # y_cords = [i*i*x1[2] + i*x1[1] + x1[0] for i in x_cords]
plt.scatter(x_cords, y_cords1, color='Blue', label='PolynomialPrediction')
plt.legend()
plt.show()

