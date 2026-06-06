import numpy as np
from sklearn.decomposition import TruncatedSVD
from config import Config

class SVDReconstructor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.svd = None
        
    def fit(self, train_X: np.ndarray) -> None:
        """Fit TruncatedSVD on training data."""
        n_components = min(self.cfg.svd_components, train_X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.cfg.seed)
        self.svd.fit(train_X)
        
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Transform, inverse transform, and compute L2 reconstruction error."""
        assert self.svd is not None, "SVDReconstructor must be fitted before computing error."
        recon = self.svd.inverse_transform(self.svd.transform(X))
        err = np.linalg.norm(X - recon, axis=1).astype(np.float32)[:, None]
        return err
