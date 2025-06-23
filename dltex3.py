import numpy as np
import matplotlib.pyplot as plt

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, lr=0.1):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(
            y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
        )
        return loss

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        dZ2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=2000):
        losses = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.3f}")
        return losses

    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        return (probs >= threshold).astype(int)

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[0],[0],[1]])
    nn = SimpleNN(input_dim=2, hidden_dim=2, lr=0.5)
    losses = nn.fit(X, y, epochs=2000)
    print("\nPredictions:")
    preds = nn.predict(X)
    for i, x in enumerate(X):
        print(f"Input: {x}, Predicted: {preds[i][0]}")
    import matplotlib.pyplot as plt
    plt.plot(range(1, 2001), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.show()
