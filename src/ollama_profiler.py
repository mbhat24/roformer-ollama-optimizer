"""
Ollama Profiler for RoPE Optimization

This module provides tools for profiling ollama's LLM inference
to identify RoPE-related bottlenecks and optimization opportunities.
"""

import subprocess
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class InferenceProfile:
    """Profile data for a single inference run."""
    model: str
    prompt: str
    prompt_tokens: int
    generation_tokens: int
    total_time_ms: float
    time_per_token_ms: float
    memory_mb: float
    cache_size_mb: float
    rope_time_ms: Optional[float] = None


class OllamaProfiler:
    """Profiler for ollama inference performance."""
    
    def __init__(self, model: str = "llama2"):
        self.model = model
        self.profiles: List[InferenceProfile] = []
    
    def run_inference(self, prompt: str, max_tokens: int = 100) -> Dict:
        """
        Run inference with ollama and profile performance.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with profiling data
        """
        # Estimate prompt tokens (rough approximation: 4 chars per token)
        prompt_tokens = len(prompt) // 4
        
        start_time = time.time()
        
        # Run ollama inference
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            generation = result.stdout
            generation_tokens = len(generation) // 4
            
        except subprocess.TimeoutExpired:
            generation = ""
            generation_tokens = 0
        except FileNotFoundError:
            print("ollama not found in PATH")
            generation = ""
            generation_tokens = 0
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if generation_tokens > 0:
            time_per_token = total_time / generation_tokens
        else:
            time_per_token = 0
        
        return {
            "model": self.model,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "total_time_ms": total_time,
            "time_per_token_ms": time_per_token,
            "generation": generation,
        }
    
    def profile_memory_usage(self) -> Dict:
        """
        Profile memory usage of ollama process.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            
            # Parse output to find ollama process
            for line in result.stdout.split('\n'):
                if 'ollama' in line.lower():
                    parts = line.split()
                    if len(parts) >= 6:
                        mem_percent = parts[3]
                        mem_mb = parts[5]
                        return {
                            "mem_percent": mem_percent,
                            "mem_mb": mem_mb,
                        }
        except Exception as e:
            print(f"Error profiling memory: {e}")
        
        return {"mem_percent": "N/A", "mem_mb": "N/A"}
    
    def profile_cache_size(self) -> Dict:
        """
        Estimate KV cache size for the current model.
        
        Returns:
            Dictionary with cache size estimates
        """
        # Rough estimation based on model size
        # This would need actual model inspection for accurate values
        model_sizes = {
            "llama2": {"d_model": 4096, "num_heads": 32, "num_layers": 32},
            "mistral": {"d_model": 4096, "num_heads": 32, "num_layers": 32},
            "gemma": {"d_model": 2048, "num_heads": 8, "num_layers": 18},
        }
        
        if self.model.lower() in model_sizes:
            specs = model_sizes[self.model.lower()]
            d_model = specs["d_model"]
            num_heads = specs["num_heads"]
            num_layers = specs["num_layers"]
            
            # KV cache size estimation (per token)
            # Each layer has 2 matrices (K, V) with d_model dimensions
            # Assuming float16 (2 bytes per element)
            cache_per_token = num_layers * 2 * d_model * 2 / (1024 ** 2)  # MB
            
            return {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "cache_per_token_mb": cache_per_token,
                "cache_1k_tokens_mb": cache_per_token * 1024,
                "cache_4k_tokens_mb": cache_per_token * 4096,
            }
        
        return {"error": "Model specs not available"}
    
    def benchmark_context_lengths(self, prompt: str, context_lengths: List[int]) -> List[Dict]:
        """
        Benchmark inference performance at different context lengths.
        
        Args:
            prompt: Base prompt
            context_lengths: List of context lengths to test
            
        Returns:
            List of profiling results
        """
        results = []
        
        for ctx_len in context_lengths:
            # Create prompt of desired length
            repeated_prompt = (prompt * (ctx_len // len(prompt) + 1))[:ctx_len]
            
            profile = self.run_inference(repeated_prompt, max_tokens=50)
            profile["context_length"] = ctx_len
            results.append(profile)
        
        return results
    
    def save_profiles(self, filename: str):
        """
        Save profiling results to JSON file.
        
        Args:
            filename: Output filename
        """
        data = [asdict(p) for p in self.profiles]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.profiles)} profiles to {filename}")
    
    def load_profiles(self, filename: str):
        """
        Load profiling results from JSON file.
        
        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.profiles = [InferenceProfile(**item) for item in data]
        print(f"Loaded {len(self.profiles)} profiles from {filename}")
    
    def generate_report(self) -> str:
        """
        Generate a summary report of profiling results.
        
        Returns:
            String with report
        """
        if not self.profiles:
            return "No profiling data available"
        
        # Calculate statistics
        avg_time_per_token = sum(p.time_per_token_ms for p in self.profiles) / len(self.profiles)
        avg_memory = sum(p.memory_mb for p in self.profiles if p.memory_mb) / len([p for p in self.profiles if p.memory_mb])
        
        report = f"""
Ollama Profiling Report
=======================
Model: {self.model}
Total Profiles: {len(self.profiles)}

Performance Metrics:
- Average Time per Token: {avg_time_per_token:.2f} ms
- Average Memory Usage: {avg_memory:.2f} MB

Individual Profiles:
"""
        
        for i, profile in enumerate(self.profiles, 1):
            report += f"""
{i}. Prompt: {profile.prompt[:50]}...
   Tokens: {profile.prompt_tokens} prompt, {profile.generation_tokens} generated
   Time: {profile.total_time_ms:.2f} ms ({profile.time_per_token_ms:.2f} ms/token)
   Memory: {profile.memory_mb} MB
"""
        
        return report


def main():
    """Run example profiling."""
    profiler = OllamaProfiler(model="llama2")
    
    # Profile cache size
    print("Cache Size Analysis:")
    cache_info = profiler.profile_cache_size()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # Run a simple inference
    print("\nRunning Inference:")
    result = profiler.run_inference("Hello, how are you?", max_tokens=50)
    for key, value in result.items():
        if key != "generation":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
