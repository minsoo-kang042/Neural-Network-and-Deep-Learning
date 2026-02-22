# Shallow Neural Network with 1 hidden layer
# The activation function is tanh(z) and sigmoid function

# Code by Minsoo Kang (Kyung hee University, Department of Mathematics)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 1. Load Dataset
X, Y = make_moons (n_samples = 1000, noise = 0.2, random_state = 42)

# 1.1 Divide Data into train / test datasets

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42
)

X_train = X_train.T             # (2,800)
X_test = X_test.T               # (2,200)
Y_train = Y_train.reshape(1,-1) # (1,800)
Y_test = Y_test.reshape(1,-1)   # (1,200)

plt.scatter(X_train[0,:], X_train[1,:], c=Y_train.ravel(), s=40, cmap = plt.cm.Spectral, edgecolors='k')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()                      # Note that it needs spiral Decision Boundary, hence logistic regression does NOT work


n_x = X_train.shape[0]                          # Input Layer (n_x): 2
n_h = 4                                         # Hidden Layer (n_h): 4
n_y = Y_train.shape[0]                          # Output Layer (n_y): 1

############################################################################################################################################
# 2 Define Basic Functions

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn (n_h, n_x) * np.sqrt(2. / n_x)          # Be aware of matrix dimension. Use the formula!
    b1 = np.zeros((n_h, 1))
    
    W2 = np.random.randn (n_y, n_h) * np.sqrt(2. / n_h)          # Note 0.01 would make weights too small (almost linear) -> He Initialization
    b2 = np.zeros((n_y,1))
    
    # Using Dictionary
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters 

def forward_propagation(X, parameters):
    # First retrieve the parameter from dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Formula for Forward Propagation
    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    # Cache Z1, A1, Z2, A2 (needed for backward propagation)
    cache = {
        "Z1": z1,
        "A1": a1,
        "Z2": z2,
        "A2": a2
    }
    
    return a2, cache
    
def compute_cost(A2, Y):
    m = Y.shape[1] # Number of samples
    cost = (-1/m) * np.sum(Y * np.log(A2 + 1e-15) + (1-Y) * np.log(1-A2 + 1e-15))    # Formula for cost function
    cost = np.squeeze(cost)
    
    return cost

def backward_propagation(parameters, cache, X, Y):                  # Calculate dZ2, dW2, db2 / dZ1, dW1, db1 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot (dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1,2))  # Derivative of tanh(x) is 1 - x^2
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims = True)
    
    # Dictionary for Gradients
    grads = {
        "dW1": dW1,
        "dW2": dW2,
        "db1": db1,
        "db2": db2
    }
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.1):
    
    # Retrieve Parameters and Gradients
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    
    # Update Rule
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    # Update the parameters dictionary
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters


def model(X,Y, n_h, num_iterations = 10000, print_cost = False):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2,Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        # For every 1000 iteration, print the cost
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
    
    return parameters

# Find the optimal W,b

parameters = model(X_train, Y_train, n_h = 4, num_iterations=10000, print_cost=True)

############################################################################################################################################
# 3. Test 

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

train_predict = predict(parameters, X_train)
test_predict = predict(parameters, X_test)

train_acc = np.mean(train_predict == Y_train) * 100
test_acc = np.mean(test_predict == Y_test) * 100


print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

############################################################################################################################################
# 4. Plot
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()].T  
    Z = model(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
    plt.ylabel('Feature 2 ($x_2$)')
    plt.xlabel('Feature 1 ($x_1$)')
    plt.show()


# Train Decision Boundary
plt.figure(figsize=(8, 6))
plt.title("Decision Boundary (Train)")
plot_decision_boundary(lambda x: predict(parameters, x), X_train, Y_train)

# Test Decision Boundary
plt.figure(figsize=(8, 6))
plt.title("Decision Boundary (Test)")
plot_decision_boundary(lambda x: predict(parameters, x), X_test, Y_test)
