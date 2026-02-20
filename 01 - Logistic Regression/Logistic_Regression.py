# Logistic Regression using Breast Cancer Dataset
# Code by Minsoo Kang (Kyung hee University, Department of Mathematics)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Data
data = load_breast_cancer()
X_raw = data.data
Y_raw = data.target

# 1.1 Divide data to Train / Test datasets
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_raw, Y_raw, test_size = 0.2, random_state = 42
)


# 1.2 Scale the Dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# 1.3 Transpose the data so that it fits as in Coursera (n_x, m)
X_train = X_train_scaled.T # (30,455)
X_test = X_test_scaled.T   # (30,114)

Y_train = Y_train_raw.reshape(1,-1)
Y_test = Y_test_raw.reshape(1,-1)       # Note that Y is rank 1 array. Hence .T (transpose) does NOT change the matrix



print(X_train.shape)   # X has 30 features with 455 train samples. 
print(Y_train.shape)   # Y tells if 0: Cancer / 1: NOT Cancer
print(X_test.shape)
print(Y_test.shape)

################################################################################################################################

# 2 Define Basic Functions

def initialize_parameters(dim):
    w = np.random.randn(dim, 1) * 0.01    # W = 30 x 1 matrix
    b = 0.0
    
    return w, b


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def forward_propagate(w, b, X, Y):
    m  = X.shape[1] # 455
    A = sigmoid(np.dot(w.T, X) + b)  # Z 계산과 sigmoid function에 대입하는 것까지 한번에
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    cost = np.squeeze(cost) # Delete unnecessary dimensions for cost
    
    return A, cost
    
    
def backward_propagate(X,Y,A):
    m = X.shape[1]
    dz = A - Y # 오차 계산: 예측값 A에서 실제 정답 Y를 빼기
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    
    return dw,db

def update_parameter(w,b,X,Y, num_iterations, learning_rate):
    # Dimensions
    # w = (30,1), b = scalar, X = (30,455), Y = (1,455)
    costs = [] # Cost Function으로 계산한 값을 저장
    
    for i in range(num_iterations):
        A, cost = forward_propagate(w,b,X,Y)
        dw, db = backward_propagate(X,Y,A)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
    
    return w, b, costs

def predict(w,b,X):
    m = X.shape[1] # 455
    Y_prediction = np.zeros((1,m))  # 결과를 담은 빈 배열
    A = 1 / (1 + np.exp(-(np.dot(w.T, X) + b)))
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
            
    return Y_prediction

################################################################################################################################

# 3 Model

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    w,b = initialize_parameters(X_train.shape[0])  # w = (30,1) matrix , b = 0.0
    w, b, costs = update_parameter(w,b,X_train,Y_train, num_iterations, learning_rate) # 최적의 w,b를 찾는 과정
    
    Y_prediction_train = predict(w,b, X_train) # 찾은 w,b로 train, test dataset을 평가
    Y_prediction_test = predict(w, b, X_test)
    
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    
    print(f"훈련 데이터 정확도: {train_accuracy:.2f}%")
    print(f"테스트 데이터 정확도: {test_accuracy:.2f}%")
    
    return w,b,costs

final_w, final_b, final_costs = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005)

################################################################################################################################

# 4 Plot

costs_to_plot = np.squeeze(final_costs)
plt.plot(costs_to_plot)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.title('Learning Rate = 0.005')
plt.show()
