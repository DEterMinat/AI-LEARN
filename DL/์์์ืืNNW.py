import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Sample data (replace with your actual data)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
T = np.array([1, 0, 1, 0])

# Train the perceptron
perceptron = Perceptron(learning_rate=0.01, max_iter=1000)  # Adjust hyperparameters as needed
perceptron.fit(X, T)

# Get the learned weights
w = perceptron.coef_[0]

# Plot the decision hyperplane using scikit-learn's visualization tools
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.plot(X, perceptron.decision_function(X), color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hyperplane for Perceptron Model')
plt.show()

print("Learned weights:", w)