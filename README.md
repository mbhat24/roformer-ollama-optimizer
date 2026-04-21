# RoFormer + Ollama Optimizer

## Overview

This project explores whether RoFormer (Rotary Position Embedding) can be used to optimize ollama's LLM performance. RoFormer introduces a novel position encoding method called RoPE (Rotary Position Embedding) that has become the default positional strategy in modern LLMs like LLaMA-2/3, Gemma, Mistral, and Code-Llama.

## Background

### What is RoFormer?

RoFormer is a transformer architecture enhancement that uses **Rotary Position Embeddings (RoPE)** to encode positional information. Key characteristics:

- **Relative by construction**: No lookup tables needed
- **Parameter-free**: Weight count identical to absolute sinusoids
- **Smooth extrapolation**: Angles extend indefinitely; with NTK/YaRN scaling, stable to 256k+ tokens
- **Streaming-friendly**: Rotation done once when writing KV cache
- **Distance-aware**: Dot product encodes only relative distance between tokens

### How RoPE Works

RoPE stores position as a rotation in each even/odd dimension pair:
- Queries and keys are rotated by R(θp,i)
- The dot-product becomes a function of (θp,i−θq,i) ∝ distance (p−q)
- Distance, not absolute index, drives attention

In simple terms: RoPE rotates query and key vectors in a shared 2-D sub-space by an angle proportional to their positions. After rotation, their dot product encodes only the relative distance.

## Current State

RoPE is **already the default** in:
- LLaMA-2/3
- Gemma
- Mistral
- Code-Llama (with NTK scaling)
- YaRN

Since ollama runs these models, RoPE is already being used. The question becomes: **how can we further optimize RoPE for ollama's performance?**

## Potential Optimization Strategies

### 1. NTK/YaRN Scaling for Longer Contexts
- Implement NTK-aware scaling for better long-context performance
- YaRN (Yet another RoPE extension) for context extension
- Test on ollama's context window limits

### 2. Custom RoPE Configurations
- Fine-tune RoPE hyperparameters for specific use cases
- Adjust theta scaling factors for domain-specific tasks
- Optimize head dimension pairings

### 3. Implementation Optimization
- Profile RoPE computation in ollama's inference path
- Optimize rotation matrix computation
- KV cache optimization for RoPE

### 4. Hybrid Approaches
- Combine RoPE with other position encoding schemes
- Adaptive RoPE for variable-length sequences
- Dynamic theta adjustment based on context

## Research Questions

1. **Can RoPE parameters be fine-tuned for specific ollama models to improve performance?**
2. **Does NTK/YaRN scaling improve ollama's long-context performance?**
3. **Can RoPE implementation be optimized for faster inference in ollama?**
4. **What is the impact of different RoPE configurations on ollama's memory usage?**

## Project Structure

```
roformer-ollama-optimizer/
├── README.md
├── docs/
│   ├── research/
│   │   ├── roformer-paper-summary.md
│   │   ├── rope-explained.md
│   │   └── ollama-architecture.md
│   └── experiments/
│       ├── baseline-performance.md
│       └── optimization-targets.md
├── src/
│   ├── rope_analysis.py
│   ├── ollama_profiler.py
│   └── optimization_experiments.py
└── tests/
    └── test_rope.py
```

## References

- RoFormer Paper: https://arxiv.org/abs/2104.09864
- RoPE Explained: https://learnopencv.com/rope-position-embeddings/
- LLaMA Nuts and Bolts: https://github.com/adalkiran/llama-nuts-and-bolts
- EleutherAI Blog on Rotary Embeddings: https://blog.eleuther.ai/rotary-embeddings/

## License

MIT License

## Contributing

This is a research project. Contributions welcome.
