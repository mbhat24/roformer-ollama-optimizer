"""
RoPE (Rotary Position Embedding) Analysis Module

This module provides tools for analyzing and optimizing RoPE implementations
for use with ollama's LLM inference.
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RoPEConfig:
    """Configuration for RoPE computation."""
    d_model: int = 512
    max_position: int = 4096
    theta_base: float = 10000.0
    scaling_type: str = "none"  # none, linear, ntk, yarn
    ntk_alpha: float = 1.0


class RoPEAnalyzer:
    """Analyzer for RoPE computation patterns and optimization opportunities."""
    
    def __init__(self, config: RoPEConfig):
        self.config = config
        self._rotation_cache: Dict[int, np.ndarray] = {}
        self._theta_cache: Optional[np.ndarray] = None
    
    def compute_theta(self, position: int) -> np.ndarray:
        """
        Compute rotation angles for a given position.
        
        Args:
            position: Position index
            
        Returns:
            Array of rotation angles for each dimension pair
        """
        d = self.config.d_model
        theta = np.zeros(d // 2)
        
        for i in range(d // 2):
            theta[i] = position * (self.config.theta_base ** (-2 * i / d))
        
        # Apply scaling if configured
        if self.config.scaling_type == "ntk":
            theta = self._apply_ntk_scaling(theta, position)
        elif self.config.scaling_type == "linear":
            theta = self._apply_linear_scaling(theta, position)
        elif self.config.scaling_type == "yarn":
            theta = self._apply_yarn_scaling(theta, position)
        
        return theta
    
    def _apply_ntk_scaling(self, theta: np.ndarray, position: int) -> np.ndarray:
        """Apply NTK-aware scaling for long contexts."""
        alpha = self.config.ntk_alpha
        # Avoid division by zero by ensuring position > 0
        position_safe = max(position, 1)
        scale = alpha * (position_safe / self.config.max_position) ** (self.config.d_model / (self.config.d_model - 2))
        # Clamp scale to avoid division by zero or extreme values
        scale = max(scale, 1e-10)
        return theta / scale
    
    def _apply_linear_scaling(self, theta: np.ndarray, position: int) -> np.ndarray:
        """Apply linear scaling."""
        # Avoid division by zero by ensuring position > 0
        position_safe = max(position, 1)
        scale = position_safe / self.config.max_position
        # Clamp scale to avoid division by zero
        scale = max(scale, 1e-10)
        return theta / scale
    
    def _apply_yarn_scaling(self, theta: np.ndarray, position: int) -> np.ndarray:
        """Apply YaRN scaling (simplified version)."""
        # YaRN is complex; this is a simplified placeholder
        scale = 1.0 + (max(position, 1) / self.config.max_position) ** 0.5
        # Clamp scale to avoid division by zero
        scale = max(scale, 1e-10)
        return theta / scale
    
    def compute_rotation_matrix(self, position: int) -> np.ndarray:
        """
        Compute 2D rotation matrix for each dimension pair.
        
        Args:
            position: Position index
            
        Returns:
            Stack of 2x2 rotation matrices for each pair
        """
        theta = self.compute_theta(position)
        num_pairs = len(theta)
        rotation_matrices = np.zeros((num_pairs, 2, 2))
        
        for i, angle in enumerate(theta):
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            rotation_matrices[i] = np.array([
                [cos_val, -sin_val],
                [sin_val, cos_val]
            ])
        
        return rotation_matrices
    
    def apply_rope(self, x: np.ndarray, position: int) -> np.ndarray:
        """
        Apply RoPE rotation to a vector.
        
        Args:
            x: Input vector of shape (d_model,)
            position: Position index
            
        Returns:
            Rotated vector
        """
        if len(x) != self.config.d_model:
            raise ValueError(f"Input vector length {len(x)} does not match d_model {self.config.d_model}")
        
        theta = self.compute_theta(position)
        x_rot = np.zeros_like(x)
        
        for i in range(self.config.d_model // 2):
            idx_even = 2 * i
            idx_odd = 2 * i + 1
            cos_val = np.cos(theta[i])
            sin_val = np.sin(theta[i])
            
            x_rot[idx_even] = x[idx_even] * cos_val - x[idx_odd] * sin_val
            x_rot[idx_odd] = x[idx_even] * sin_val + x[idx_odd] * cos_val
        
        return x_rot
    
    def apply_rope_batch(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Apply RoPE rotation to a batch of vectors.
        
        Args:
            x: Input vectors of shape (batch_size, d_model)
            positions: Position indices of shape (batch_size,)
            
        Returns:
            Rotated vectors
        """
        batch_size = x.shape[0]
        x_rot = np.zeros_like(x)
        
        for b in range(batch_size):
            x_rot[b] = self.apply_rope(x[b], positions[b])
        
        return x_rot
    
    def precompute_rotations(self, max_position: int):
        """
        Pre-compute and cache rotation matrices for positions up to max_position.
        
        Args:
            max_position: Maximum position to pre-compute
        """
        for pos in range(max_position):
            self._rotation_cache[pos] = self.compute_rotation_matrix(pos)
    
    def apply_rope_cached(self, x: np.ndarray, position: int) -> np.ndarray:
        """
        Apply RoPE rotation using cached rotation matrices.
        
        Args:
            x: Input vector of shape (d_model,)
            position: Position index
            
        Returns:
            Rotated vector
        """
        if position not in self._rotation_cache:
            # Fallback to computation if not cached
            return self.apply_rope(x, position)
        
        rotation_matrices = self._rotation_cache[position]
        x_rot = np.zeros_like(x)
        
        for i in range(self.config.d_model // 2):
            idx_even = 2 * i
            idx_odd = 2 * i + 1
            rot = rotation_matrices[i]
            
            x_rot[idx_even] = x[idx_even] * rot[0, 0] - x[idx_odd] * rot[0, 1]
            x_rot[idx_odd] = x[idx_even] * rot[1, 0] + x[idx_odd] * rot[1, 1]
        
        return x_rot
    
    def analyze_angle_distribution(self, max_position: int = 100) -> Dict:
        """
        Analyze the distribution of rotation angles across positions.
        
        Args:
            max_position: Maximum position to analyze
            
        Returns:
            Dictionary with analysis results
        """
        angles = []
        for pos in range(max_position):
            theta = self.compute_theta(pos)
            angles.extend(theta)
        
        angles = np.array(angles)
        
        return {
            "mean": np.mean(angles),
            "std": np.std(angles),
            "min": np.min(angles),
            "max": np.max(angles),
            "median": np.median(angles),
            "num_oscillations": np.sum(angles > np.pi),
        }
    
    def compute_attention_decay(self, query_pos: int, key_pos: int, pair_idx: int = 0) -> float:
        """
        Compute the attention decay based on positional distance.
        
        Args:
            query_pos: Query position
            key_pos: Key position
            pair_idx: Dimension pair index
            
        Returns:
            Cosine of angle difference (attention weight proxy)
        """
        theta_q = self.compute_theta(query_pos)[pair_idx]
        theta_k = self.compute_theta(key_pos)[pair_idx]
        return np.cos(theta_q - theta_k)
    
    def profile_computation_time(self, num_iterations: int = 1000) -> Dict:
        """
        Profile RoPE computation time.
        
        Args:
            num_iterations: Number of iterations to average over
            
        Returns:
            Dictionary with timing results
        """
        import time
        
        # Test vector
        x = np.random.randn(self.config.d_model)
        positions = np.random.randint(0, self.config.max_position, num_iterations)
        
        # Time standard computation
        start = time.time()
        for i in range(num_iterations):
            self.apply_rope(x, positions[i])
        standard_time = time.time() - start
        
        # Pre-compute and cache
        self.precompute_rotations(self.config.max_position)
        
        # Time cached computation
        start = time.time()
        for i in range(num_iterations):
            self.apply_rope_cached(x, positions[i])
        cached_time = time.time() - start
        
        speedup = standard_time / cached_time
        
        return {
            "standard_time_ms": standard_time * 1000 / num_iterations,
            "cached_time_ms": cached_time * 1000 / num_iterations,
            "speedup": speedup,
            "cache_size_mb": len(self._rotation_cache) * self.config.d_model * 8 / (1024 ** 2),
        }


class RoPEVisualizer:
    """Visualization tools for RoPE analysis."""
    
    def __init__(self, analyzer: RoPEAnalyzer):
        self.analyzer = analyzer
    
    def plot_angle_vs_position(self, max_position: int = 100):
        """
        Plot rotation angles vs position for different dimension pairs.
        
        Args:
            max_position: Maximum position to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        positions = np.arange(max_position)
        num_pairs_to_plot = min(5, self.analyzer.config.d_model // 2)
        
        plt.figure(figsize=(12, 6))
        
        for pair_idx in range(num_pairs_to_plot):
            angles = []
            for pos in positions:
                theta = self.analyzer.compute_theta(pos)
                angles.append(theta[pair_idx])
            
            plt.plot(positions, angles, label=f'Pair {pair_idx}')
        
        plt.xlabel('Position')
        plt.ylabel('Rotation Angle (radians)')
        plt.title('RoPE Rotation Angles vs Position')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt
    
    def plot_attention_decay(self, query_pos: int = 50, max_distance: int = 50):
        """
        Plot attention decay as function of positional distance.
        
        Args:
            query_pos: Query position
            max_distance: Maximum distance to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        distances = np.arange(max_distance)
        num_pairs_to_plot = min(5, self.analyzer.config.d_model // 2)
        
        plt.figure(figsize=(12, 6))
        
        for pair_idx in range(num_pairs_to_plot):
            decays = []
            for dist in distances:
                key_pos = query_pos + dist
                decay = self.analyzer.compute_attention_decay(query_pos, key_pos, pair_idx)
                decays.append(decay)
            
            plt.plot(distances, decays, label=f'Pair {pair_idx}')
        
        plt.xlabel('Positional Distance')
        plt.ylabel('Attention Weight (cos of angle difference)')
        plt.title(f'RoPE Attention Decay (Query Position={query_pos})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt


def main():
    """Run example RoPE analysis."""
    config = RoPEConfig(d_model=512, max_position=4096, scaling_type="none")
    analyzer = RoPEAnalyzer(config)
    
    # Analyze angle distribution
    print("Angle Distribution Analysis:")
    analysis = analyzer.analyze_angle_distribution(100)
    for key, value in analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # Profile computation time
    print("\nComputation Time Profiling:")
    profile = analyzer.profile_computation_time(1000)
    for key, value in profile.items():
        print(f"  {key}: {value:.4f}")
    
    # Test attention decay
    print("\nAttention Decay Examples:")
    for dist in [0, 1, 5, 10, 20, 50]:
        decay = analyzer.compute_attention_decay(0, dist, 0)
        print(f"  Distance {dist:2d}: {decay:.4f}")


if __name__ == "__main__":
    main()
