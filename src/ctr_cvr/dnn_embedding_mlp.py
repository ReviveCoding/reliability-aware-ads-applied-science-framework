import numpy as np

class TorchEmbeddingMLP:
    '''
    Optional lightweight PyTorch DNN model for smoke/sample runs.

    The pipeline remains runnable without this class being trained. It is provided
    as a JD-aligned DNN implementation hook.
    '''
    def __init__(self, epochs=3, lr=1e-3, seed=42):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.available = False
        try:
            import torch
            self.torch = torch
            self.available = True
        except Exception:
            self.torch = None

    def fit(self, X, y):
        if not self.available:
            raise RuntimeError("PyTorch is not available.")
        torch = self.torch
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.mean_) / self.std_
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(Xn.shape[1], 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCELoss()
        Xt = torch.tensor(Xn, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model_(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        if not self.available:
            raise RuntimeError("PyTorch is not available.")
        torch = self.torch
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mean_) / self.std_
        with torch.no_grad():
            pred = self.model_(torch.tensor(Xn, dtype=torch.float32)).numpy().reshape(-1)
        return np.clip(pred, 1e-6, 1 - 1e-6)
