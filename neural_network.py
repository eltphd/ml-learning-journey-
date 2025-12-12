"""
neural_network.py
Simple neural network implementation from scratch using only NumPy
Author: Erica L. Tartt, PhD
Date: 2024

Learning goals:
- Understand forward propagation mechanics
- Implement backpropagation from first principles
- Grasp gradient descent optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    """
    Multi-layer perceptron with one hidden layer
    
    Architecture:
        Input -> Hidden (ReLU) -> Output (Sigmoid)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize network with random weights
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Step size for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize weights with Xavier initialization
        # Helps prevent vanishing/exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Track loss history for visualization
        self.loss_history = []
        
    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid activation: 1 / (1 + exp(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, a):
        """Derivative of sigmoid (in terms of output)"""
        return a * (1 - a)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            a2: Output predictions (n_samples, n_outputs)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        """
        Backpropagation to compute gradients
        
        Args:
            X: Input data (n_samples, n_features)
            y: True labels (n_samples, n_outputs)
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        # dL/dz2 = (a2 - y) * sigmoid'(z2)
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        # dL/dz1 = (dL/dz2 * W2^T) * relu'(z1)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss
        
        L = -[y*log(y_pred) + (1-y)*log(1-y_pred)]
        """
        m = y_true.shape[0]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the network
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples, 1)
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            accuracy: Proportion of correct predictions
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def plot_decision_boundary(model, X, y):
    """
    Visualize the decision boundary learned by the model
    """
    # Create mesh grid
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundary')
    plt.colorbar()
    plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Decision boundary plot saved to: decision_boundary.png")


def main():
    """
    Test the neural network on a toy dataset
    """
    print("=" * 60)
    print("Neural Network from Scratch - Demo")
    print("=" * 60)
    
    # Generate synthetic dataset (two moons - non-linearly separable)
    print("\n1. Generating synthetic dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)  # Reshape for network compatibility
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Create and train network
    print("\n2. Training neural network...")
    nn = NeuralNetwork(
        input_size=2,
        hidden_size=10,
        output_size=1,
        learning_rate=0.1
    )
    
    nn.train(X_train, y_train, epochs=500, verbose=True)
    
    # Evaluate on test set
    print("\n3. Evaluating on test set...")
    test_accuracy = nn.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Plot learning curve
    print("\n4. Plotting learning curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Learning curve saved to: learning_curve.png")
    
    # Plot decision boundary
    print("\n5. Visualizing decision boundary...")
    plot_decision_boundary(nn, X, y)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
