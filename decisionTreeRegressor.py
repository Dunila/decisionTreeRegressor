from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import json

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: int = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2
    n_features_: int = None
    tree_: Node = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a two given sets of target values"""
        mse_left, mse_right = self._mse(y_left), self._mse(y_right)
        n_left, n_right = len(y_left), len(y_right)
        return (mse_left * n_left + mse_right * n_right) / (n_left + n_right)

    def _splited_samples(self, X: np.ndarray, y: np.ndarray, feature:int, threshold: int):
        return y[np.where(X[:, feature] <= threshold)], y[np.where(X[:, feature] > threshold)]

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int) -> float:
        """Find the best split for a node (one feature)"""
        x_f = X[:, feature]
        thresholds = np.unique(x_f)[1:-2]
        best_threshold = 0
        best_crit = np.inf
        for threshold in thresholds:
            left_part, right_part = y[np.where(x_f <= threshold)], y[np.where(x_f > threshold)]
            current_crit = self._weighted_mse(left_part, right_part)
            if current_crit < best_crit:
                best_crit = current_crit
                best_threshold = threshold
        return best_threshold

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        min_mse = np.inf
        best_feature, best_threshold = -1, -1
        for feature in range(X.shape[1]):
            best_split_on_feature = self._split(X, y, feature)
            wmse_on_feature = self._weighted_mse(
                                    y[np.where(X[:, feature] <= best_split_on_feature)],
                                    y[np.where(X[:, feature] > best_split_on_feature)]
                                )
            if wmse_on_feature < min_mse:
                best_feature = feature
                best_threshold = best_split_on_feature
                min_mse = wmse_on_feature
        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        node = Node(n_samples=len(y))
        if depth < self.max_depth and node.n_samples > self.min_samples_split:
            best_feature, best_threshold = self._best_split(X, y)
            node.feature = best_feature
            node.threshold = best_threshold
            x_left, x_right = X[np.where(X[:, best_feature] <= best_threshold)], X[np.where(X[:, best_feature] > best_threshold)]
            y_left, y_right = y[np.where(X[:, best_feature] <= best_threshold)], y[np.where(X[:, best_feature] > best_threshold)]
            node.value = int(np.round(np.mean(y)))
            node.mse = self._mse(y)
            node.left = self._split_node(x_left, y_left, depth+1)
            node.right = self._split_node(x_right, y_right, depth+1)
        else:
            node.value = int(np.round(np.mean(y)))
            node.mse = self._mse(y)
        return node

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return json.dumps(self._as_json(self.tree_))

    def _as_json1(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left and node.right:
            left = self._as_json(node.left)
            right = self._as_json(node.right)
            return f'"threshold": {node.threshold}, "feature": {node.feature}, "mse": {np.round(node.mse, 2)}, "n_samples": {node.n_samples}, "right": {{{right}}}, "left": {{{left}}}'
        elif node.left and not node.right:
            left = self._as_json(node.left)
            return f'"threshold": {node.threshold}, "feature": {node.feature}, "mse": {np.round(node.mse, 2)}, "n_samples": {node.n_samples}, "left": {{{left}}}'
        elif not node.left and node.right:
            right = self._as_json(node.right)
            return f'"threshold": {node.threshold}, "feature": {node.feature}, "mse": {np.round(node.mse, 2)}, "n_samples": {node.n_samples}, "right": {{{right}}}'
        else:
            return f'"value": {str(node.value)}, "n_samples": {str(node.n_samples)}, "mse": {np.round(node.mse, 2)}'

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left and node.right:
            left = self._as_json(node.left)
            right = self._as_json(node.right)
            return {"threshold": int(node.threshold), 
                    "feature": int(node.feature), 
                    "mse": float(np.round(node.mse, 2)), 
                    "n_samples": int(node.n_samples), 
                    "right": right, 
                    "left": left}
        elif node.left and not node.right:
            left = self._as_json(node.left)
            return {"threshold": int(node.threshold), 
                    "feature": int(node.feature), 
                    "mse": float(np.round(node.mse, 2)), 
                    "n_samples": int(node.n_samples),
                    "left": left}
        elif not node.left and node.right:
            right = self._as_json(node.right)
            return {"threshold": int(node.threshold), 
                    "feature": int(node.feature), 
                    "mse": float(np.round(node.mse, 2)), 
                    "n_samples": int(node.n_samples), 
                    "right": right}
        else:
            return {"value": int(node.value), 
                    "n_samples": int(node.n_samples), 
                    "mse": float(np.round(node.mse, 2))
                    }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        return np.apply_along_axis(self._predict_one_sample, 1, X)


    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        if self.tree_:
            node = self.tree_
        else:
            raise ValueError("Not fitted")
        while node.left and node.right:
            if features[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
