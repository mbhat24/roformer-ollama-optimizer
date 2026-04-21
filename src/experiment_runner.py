"""
Experiment Runner for RoPE Optimization Research

This module runs comprehensive experiments to generate data for publication.
"""

import numpy as np
import json
import time
from typing import Dict, List
from rope_analysis import RoPEAnalyzer, RoPEConfig


class ExperimentRunner:
    """Runner for RoPE optimization experiments."""
    
    def __init__(self):
        self.results = {}
    
    def run_baseline_experiments(self):
        """Run baseline experiments with standard RoPE configuration."""
        print("Running baseline experiments...")
        
        config = RoPEConfig(d_model=512, max_position=4096, scaling_type="none")
        analyzer = RoPEAnalyzer(config)
        
        # Angle distribution analysis
        angle_dist = analyzer.analyze_angle_distribution(100)
        
        # Computation profiling
        profile = analyzer.profile_computation_time(1000)
        
        # Attention decay at different distances
        decay_data = {}
        for dist in [0, 1, 5, 10, 20, 50, 100]:
            decay_data[f"dist_{dist}"] = analyzer.compute_attention_decay(0, dist, 0)
        
        self.results["baseline"] = {
            "config": {"d_model": 512, "max_position": 4096, "scaling": "none"},
            "angle_distribution": angle_dist,
            "computation_profile": profile,
            "attention_decay": decay_data,
        }
        
        print("Baseline experiments completed.")
    
    def run_scaling_comparison(self):
        """Compare different scaling methods."""
        print("Running scaling comparison experiments...")
        
        scaling_types = ["none", "linear", "ntk"]
        d_model = 512
        max_position = 4096
        
        for scaling in scaling_types:
            config = RoPEConfig(d_model=d_model, max_position=max_position, scaling_type=scaling)
            analyzer = RoPEAnalyzer(config)
            
            # Profile at long context
            profile = analyzer.profile_computation_time(1000)
            
            # Analyze angle distribution at long context
            angle_dist = analyzer.analyze_angle_distribution(100)
            
            self.results[f"scaling_{scaling}"] = {
                "config": {"d_model": d_model, "max_position": max_position, "scaling": scaling},
                "computation_profile": profile,
                "angle_distribution": angle_dist,
            }
        
        print("Scaling comparison completed.")
    
    def run_d_model_comparison(self):
        """Compare performance across different model sizes."""
        print("Running d_model comparison experiments...")
        
        d_models = [256, 512, 768, 1024, 2048]
        
        for d_model in d_models:
            config = RoPEConfig(d_model=d_model, max_position=4096, scaling_type="none")
            analyzer = RoPEAnalyzer(config)
            
            # Profile computation
            profile = analyzer.profile_computation_time(1000)
            
            # Cache size estimation
            analyzer.precompute_rotations(4096)
            cache_size = len(analyzer._rotation_cache) * d_model * 8 / (1024 ** 2)
            
            self.results[f"d_model_{d_model}"] = {
                "config": {"d_model": d_model, "max_position": 4096, "scaling": "none"},
                "computation_profile": profile,
                "cache_size_mb": cache_size,
            }
        
        print("d_model comparison completed.")
    
    def run_context_length_comparison(self):
        """Compare performance at different context lengths."""
        print("Running context length comparison experiments...")
        
        context_lengths = [1024, 2048, 4096, 8192]
        d_model = 512
        
        for ctx_len in context_lengths:
            config = RoPEConfig(d_model=d_model, max_position=ctx_len, scaling_type="ntk")
            analyzer = RoPEAnalyzer(config)
            
            # Profile computation
            profile = analyzer.profile_computation_time(1000)
            
            # Analyze angle distribution
            angle_dist = analyzer.analyze_angle_distribution(min(ctx_len, 100))
            
            self.results[f"context_{ctx_len}"] = {
                "config": {"d_model": d_model, "max_position": ctx_len, "scaling": "ntk"},
                "computation_profile": profile,
                "angle_distribution": angle_dist,
            }
        
        print("Context length comparison completed.")
    
    def run_cache_hit_rate_analysis(self):
        """Analyze cache hit rate for pre-computed rotations."""
        print("Running cache hit rate analysis...")
        
        config = RoPEConfig(d_model=512, max_position=4096, scaling_type="none")
        analyzer = RoPEAnalyzer(config)
        
        # Pre-compute for different cache sizes
        cache_sizes = [256, 512, 1024, 2048, 4096]
        
        for cache_size in cache_sizes:
            analyzer.precompute_rotations(cache_size)
            
            # Simulate access pattern (random positions)
            num_accesses = 10000
            positions = np.random.randint(0, 4096, num_accesses)
            
            hits = sum(1 for pos in positions if pos in analyzer._rotation_cache)
            hit_rate = hits / num_accesses
            
            # Time with cache
            x = np.random.randn(512)
            start = time.time()
            for pos in positions[:1000]:
                if pos in analyzer._rotation_cache:
                    analyzer.apply_rope_cached(x, pos)
            cached_time = time.time() - start
            
            # Time without cache
            start = time.time()
            for pos in positions[:1000]:
                analyzer.apply_rope(x, pos)
            standard_time = time.time() - start
            
            self.results[f"cache_{cache_size}"] = {
                "cache_size": cache_size,
                "hit_rate": hit_rate,
                "cached_time_ms": cached_time * 1000,
                "standard_time_ms": standard_time * 1000,
                "speedup": standard_time / cached_time,
            }
        
        print("Cache hit rate analysis completed.")
    
    def run_all_experiments(self):
        """Run all experiments and collect data."""
        print("Starting comprehensive experiment suite...")
        
        self.run_baseline_experiments()
        self.run_scaling_comparison()
        self.run_d_model_comparison()
        self.run_context_length_comparison()
        self.run_cache_hit_rate_analysis()
        
        print("All experiments completed.")
    
    def save_results(self, filename: str):
        """Save experimental results to JSON file."""
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_native = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_native, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def generate_summary(self) -> str:
        """Generate a summary of experimental results."""
        summary = "Experimental Results Summary\n"
        summary += "=" * 50 + "\n\n"
        
        # Baseline results
        if "baseline" in self.results:
            baseline = self.results["baseline"]
            summary += "Baseline Configuration (d_model=512, scaling=none):\n"
            summary += f"  Standard computation: {baseline['computation_profile']['standard_time_ms']:.4f} ms/token\n"
            summary += f"  Cached computation: {baseline['computation_profile']['cached_time_ms']:.4f} ms/token\n"
            summary += f"  Speedup: {baseline['computation_profile']['speedup']:.2f}x\n"
            summary += f"  Cache size: {baseline['computation_profile']['cache_size_mb']:.2f} MB\n"
            summary += f"  Angle oscillations: {baseline['angle_distribution']['num_oscillations']:.0f}\n\n"
        
        # Scaling comparison
        summary += "Scaling Method Comparison:\n"
        for scaling in ["none", "linear", "ntk"]:
            key = f"scaling_{scaling}"
            if key in self.results:
                result = self.results[key]
                summary += f"  {scaling}: {result['computation_profile']['standard_time_ms']:.4f} ms/token\n"
        summary += "\n"
        
        # d_model comparison
        summary += "Model Size Comparison:\n"
        for d_model in [256, 512, 1024, 2048]:
            key = f"d_model_{d_model}"
            if key in self.results:
                result = self.results[key]
                summary += f"  d_model={d_model}: {result['computation_profile']['standard_time_ms']:.4f} ms/token, cache={result['cache_size_mb']:.2f} MB\n"
        summary += "\n"
        
        # Cache analysis
        summary += "Cache Performance:\n"
        for cache_size in [256, 512, 1024, 2048, 4096]:
            key = f"cache_{cache_size}"
            if key in self.results:
                result = self.results[key]
                summary += f"  Cache {cache_size}: hit_rate={result['hit_rate']:.2%}, speedup={result['speedup']:.2f}x\n"
        
        return summary


def main():
    """Run all experiments and save results."""
    runner = ExperimentRunner()
    
    runner.run_all_experiments()
    runner.save_results("results/experimental_results.json")
    
    print("\n" + runner.generate_summary())


if __name__ == "__main__":
    main()
