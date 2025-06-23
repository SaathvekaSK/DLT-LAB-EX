import numpy as np

class LogisticRegressionNN:
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)

    def compute_loss(self, y_true, y_pred_prob):
        m = len(y_true)
        loss = - (1/m) * np.sum(y_true * np.log(y_pred_prob + 1e-15) + (1 - y_true) * np.log(1 - y_pred_prob + 1e-15))
        return loss

    def fit(self, X, y, epochs=5000):
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            y_pred_prob = self.predict_prob(X)
            loss = self.compute_loss(y, y_pred_prob)

            dw = (1/m) * np.dot(X.T, (y_pred_prob - y))
            db = (1/m) * np.sum(y_pred_prob - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                print(f"Iteration {epoch}/{epochs}, Cost: {loss:.4f}")

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])

    model = LogisticRegressionNN(input_dim=2, lr=0.1)
    model.fit(X, y, epochs=5000)

    print("\nPredictions:")
    for x_sample in X:
        pred = model.predict(x_sample.reshape(1, -1))[0]
        print(f"Input: {x_sample}, Predicted: {pred}")
