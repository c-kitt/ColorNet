import cmd
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1  / (1 + np.exp(-x))

class color_net:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(0, np.sqrt(2/3), size=(3, 3))
        self.b1 = np.zeros(3)

        self.w2 = rng.normal(0, np.sqrt(1/3), size=(1, 3))
        self.b2 = np.zeros(1)

    # Forward Propagation
    def forward(self, rgb):
        # Input Layer, scale to [0, 1]
        x = np.asarray(rgb, dtype=float) / 255.0

        # Hidden Layer: linear -> ReLU
        h_pre = self.w1 @ x + self.b1
        h = relu(h_pre)

        # Output layer: linear -> sigmoid
        z = self.w2 @ h + self.b2
        p = sigmoid(z)[0]

        cache = {"x": x, "h_pre": h_pre, "h": h, "z": z, "p": p}
        return p, cache

    # Backward Propagation
    def backward(self, cache, y):
        x = cache["x"]        
        h_pre = cache["h_pre"]
        h = cache["h"]     
        z = cache["z"]     
        p = cache["p"]        

        # dL/dz
        dz = p - float(y)

        # Output layer gradients
        dw2 = dz * h.reshape(1, -1)
        db2 = np.array([dz])

        # Back-prop into hidden: dL/dh = dz * W2^T
        gh = dz * self.w2.flatten()

        # ReLU: derivative is 1, where h_pre > 0, else 0.
        relu_mask = (h_pre > 0).astype(float)
        gh_pre = gh * relu_mask

        # First layer gradients
        dw1 = gh_pre.reshape(-1,1) @ x.reshape(1, -1)
        db1 = gh_pre

        return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

    @staticmethod
    def cross_entropy_loss(p, y, eps = 1e-12):
        p = float(p)
        y = float(y)
        p = min(max(p, eps), 1 - eps) # Avoid log(0)

        # Binary cross-entropy function
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def train_step(self, rgb, y, learn_rate = 0.1):
        # Forward Pass
        p, cache = self.forward(rgb)

        # Compute Loss
        loss = self.cross_entropy_loss(p, y)

        # Backward Pass
        gradients = self.backward(cache, y)
        dw1, db1, dw2, db2 = gradients["dw1"], gradients["db1"], gradients["dw2"], gradients["db2"]

        # Gradient Descent
        self.w1 -= learn_rate * dw1
        self.b1 -= learn_rate * db1
        self.w2 -= learn_rate * dw2
        self.b2 -= learn_rate * db2

        return loss, p

    def predict_proba(self, rgb):
        p, _ = self.forward(rgb)
        return p

    def predict(self, rgb, threshold=0.5):
        p = self.predict_proba(rgb)
        return 1 if p >= threshold else 0

    def state_dict(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def save(self, path:str):
        np.savez(path, **self.state_dict(), version=1)
    
    def load_state(self, path: str):
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]

    @classmethod
    def load(cls, path: str):
        obj = cls(seed=0)
        obj.load_state(path)
        return obj

if __name__ == "__main__":
    np.random.seed(0)
    cn = color_net(seed=42)

    def brightness_label(rgb):
        r, g, b = np.array(rgb, dtype=float) / 255.0
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return 1 if y < 0.5 else 0  

    # Train 
    steps = 200000
    lr = 0.001
    for t in range(1, steps + 1):
        rgb = np.random.randint(0, 256, size=3)
        y = brightness_label(rgb)
        loss, p = cn.train_step(rgb, y, learn_rate=lr)
        if t % 20000 == 0:
            print(f"Step {t:4d} | loss={loss:.4f} | p={p:.3f}")

    # Evaluate 
    N = 200000
    correct = 0
    for _ in range(N):
        rgb = np.random.randint(0, 256, size=3)
        y_true = brightness_label(rgb)
        y_pred = cn.predict(rgb)
        correct += (y_pred == y_true)

    acc = correct / N
    print(f"Validation accuracy: {acc:.3f}")

    cn.save("color_net.npz")
    print("Saved weights to color_net.npz")