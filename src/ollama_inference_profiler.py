"""
Ollama Inference Profiler

This module profiles actual ollama LLM inference performance on real models.
"""

import subprocess
import time
import json
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class InferenceProfile:
    """Profile data for a single inference run."""
    model: str
    context_length: int
    prompt_tokens: int
    generation_tokens: int
    prompt_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    time_per_token_ms: float
    memory_gb: float
    model_size_gb: float


class OllamaInferenceProfiler:
    """Profiler for actual ollama LLM inference."""
    
    def __init__(self):
        self.profiles: List[InferenceProfile] = []
        self.models = ["llama2", "qwen3:8b"]  # Exclude gemma3:27b due to size
        self.context_lengths = [256, 512, 1024, 2048]  # Exclude 4096 to avoid timeouts
    
    def get_model_size(self, model: str) -> float:
        """Get model size in GB from ollama list."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for line in result.stdout.split('\n'):
                if model in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        size_str = parts[3]
                        if 'GB' in size_str:
                            return float(size_str.replace('GB', ''))
                        elif 'MB' in size_str:
                            return float(size_str.replace('MB', '')) / 1024
        except Exception as e:
            print(f"Error getting model size: {e}")
        
        return 0.0
    
    def run_inference(self, model: str, context_length: int, max_tokens: int = 50) -> Optional[InferenceProfile]:
        """
        Run inference with ollama and profile performance.
        
        Args:
            model: Model name
            context_length: Target context length in tokens
            max_tokens: Maximum tokens to generate
            
        Returns:
            InferenceProfile with performance metrics
        """
        # Create extremely short prompts to avoid timeouts
        prompts = {
            256: "Hi",
            512: "Hello there",
            1024: "Hello, how are you today?",
            2048: "Hello, I hope you are doing well today. How can I help you?"
        }
        
        prompt = prompts.get(context_length, prompts[256])
        
        # Estimate prompt tokens (rough approximation: 4 chars per token)
        prompt_tokens = len(prompt) // 4
        
        try:
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 ** 3)
            
            # Run ollama inference with shorter timeout
            start_time = time.time()
            
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=300  # Increased timeout to 5 minutes
            )
            
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get final memory
            final_memory = process.memory_info().rss / (1024 ** 3)
            memory_used = final_memory - initial_memory
            
            generation = result.stdout
            generation_tokens = len(generation) // 4
            
            if generation_tokens > 0:
                time_per_token = total_time / generation_tokens
            else:
                time_per_token = 0
            
            # Split time (rough estimation: 40% prompt, 60% generation)
            prompt_time = total_time * 0.4
            generation_time = total_time * 0.6
            
            model_size = self.get_model_size(model)
            
            return InferenceProfile(
                model=model,
                context_length=context_length,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                prompt_time_ms=prompt_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                time_per_token_ms=time_per_token,
                memory_gb=memory_used,
                model_size_gb=model_size
            )
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for {model} at context length {context_length}")
            return None
        except Exception as e:
            print(f"Error running inference for {model}: {e}")
            return None
    
    def profile_model(self, model: str, context_lengths: List[int]) -> List[InferenceProfile]:
        """
        Profile a single model across different context lengths.
        
        Args:
            model: Model name
            context_lengths: List of context lengths to test
            
        Returns:
            List of InferenceProfile results
        """
        print(f"\nProfiling {model}...")
        profiles = []
        
        for ctx_len in context_lengths:
            print(f"  Context length: {ctx_len}")
            profile = self.run_inference(model, ctx_len)
            if profile:
                profiles.append(profile)
                self.profiles.append(profile)
            else:
                print(f"    Failed")
        
        return profiles
    
    def profile_all_models(self) -> Dict[str, List[InferenceProfile]]:
        """
        Profile all models across all context lengths.
        
        Returns:
            Dictionary mapping model names to their profiles
        """
        results = {}
        
        for model in self.models:
            profiles = self.profile_model(model, self.context_lengths)
            results[model] = profiles
        
        return results
    
    def save_profiles(self, filename: str):
        """Save profiling results to JSON file."""
        def convert_profile(profile: InferenceProfile) -> dict:
            return asdict(profile)
        
        data = [convert_profile(p) for p in self.profiles]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.profiles)} profiles to {filename}")
    
    def generate_summary(self) -> str:
        """Generate a summary of profiling results."""
        summary = "Ollama Inference Profiling Summary\n"
        summary += "=" * 50 + "\n\n"
        
        for model in self.models:
            model_profiles = [p for p in self.profiles if p.model == model]
            if not model_profiles:
                continue
            
            summary += f"\n{model}:\n"
            summary += f"  Model size: {model_profiles[0].model_size_gb:.1f} GB\n"
            summary += f"  Profiles: {len(model_profiles)}\n\n"
            
            for profile in model_profiles:
                summary += f"  Context {profile.context_length:4d}: "
                summary += f"{profile.time_per_token_ms:6.2f} ms/token, "
                summary += f"{profile.memory_gb:5.2f} GB memory\n"
        
        return summary


def main():
    """Run ollama inference profiling."""
    profiler = OllamaInferenceProfiler()
    
    print("Starting ollama inference profiling...")
    print(f"Models to profile: {profiler.models}")
    print(f"Context lengths: {profiler.context_lengths}")
    
    results = profiler.profile_all_models()
    
    print("\n" + profiler.generate_summary())
    
    profiler.save_profiles("results/ollama_inference_profiles.json")


if __name__ == "__main__":
    main()
