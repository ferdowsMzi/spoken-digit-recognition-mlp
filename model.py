import cupy as np

class MLP:
    def __init__(self, layer_sizes, activation='bipolar_sigmoid', lr=0.001):
        self.sizes = layer_sizes
        self.lr = lr
        self.activation_name = activation

        self.W = [np.random.randn(self.sizes[i], self.sizes[i+1]) * np.sqrt(2. / self.sizes[i]) for i in range(len(self.sizes)-1)] #He initialization
        self.b = [np.zeros((1, self.sizes[i+1])) for i in range(len(self.sizes)-1)]

    
    def _activate(self, x, is_output_layer=False):
        if self.activation_name == 'relu' and not is_output_layer:
            return np.maximum(0, x)
        return (1 - np.exp(-x)) / (1 + np.exp(-x))
    
    
    def _derivative(self, a, is_output_layer=False):
        if self.activation_name == 'relu' and not is_output_layer:
            return (a > 0).astype(float)
        return 0.5 * (1 - a**2)
    

    def forward(self, x):
        self.a = [np.array(x, ndmin=2)]

        for i in range(len(self.W)):
            z = np.dot(self.a[-1], self.W[i]) + self.b[i]
            is_output = (i == len(self.W) - 1)
            a = self._activate(z, is_output_layer=is_output)
            self.a.append(a)

        return self.a[-1]
    
    def backward(self, y_true):
        y_true = np.array(y_true, ndmin=2)
        y_pred = self.a[-1]

        error = y_pred - y_true
        delta = error * self._derivative(y_pred, is_output_layer=True)

        for i in reversed(range(len(self.W))):
            dW = np.dot(self.a[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            deriv = self._derivative(self.a[i], is_output_layer=False)
            delta = np.dot(delta, self.W[i].T) * deriv

            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db


    def compute_mse(self, X, y_one_hot):
        z = X
        for i in range(len(self.W)):
            z = self._activate(np.dot(z, self.W[i]) + self.b[i], is_output_layer=(i == len(self.W)-1))
        return np.mean((z - y_one_hot) ** 2)
    
    def predict(self, X):
        z = X
        for i in range(len(self.W)):
            z = self._activate(np.dot(z, self.W[i]) + self.b[i], is_output_layer=(i == len(self.W)-1))
        return np.argmax(z, axis=1)