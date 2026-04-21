# RoPE (Rotary Position Embedding) Explained

## What is RoPE?

Rotary Position Embedding (RoPE) is a position encoding technique for transformers that encodes positional information by rotating query and key vectors in a shared 2-D sub-space.

## Core Concept

### Traditional Position Encodings

**Absolute Position Encoding** (e.g., sinusoidal, learned embeddings):
- Each position has a fixed embedding
- No notion of relative distance
- Limited extrapolation to longer sequences

**Relative Position Encoding** (e.g., T5, ALiBi):
- Explicitly encodes relative distances
- Requires lookup tables
- Can be computationally expensive

### RoPE Innovation

RoPE combines the best of both:
- **Relative by construction**: Distance emerges from rotation
- **Parameter-free**: No lookup tables needed
- **Smooth extrapolation**: Angles extend indefinitely

## How RoPE Works

### Step 1: Pair Dimensions

For a hidden dimension d (must be even), split into d/2 pairs:
- Pair 0: dimensions (0, 1)
- Pair 1: dimensions (2, 3)
- Pair k: dimensions (2k, 2k+1)

### Step 2: Compute Rotation Angle

For position p and pair index k:
```
θ(p, k) = p * ω_k
where ω_k = 1 / (10000^(2k/d))
```

This is often written as:
```
θ(p, k) = p^(-2k/d)
```

### Step 3: Rotation Matrix

For each pair, apply 2D rotation:
```
R(θ) = [[cos θ, -sin θ],
         [sin θ,  cos θ]]
```

### Step 4: Rotate Queries and Keys

For each position p and pair k:
```
q_rot[2k]   = q[2k]   * cos(θ) - q[2k+1] * sin(θ)
q_rot[2k+1] = q[2k]   * sin(θ) + q[2k+1] * cos(θ)

k_rot[2k]   = k[2k]   * cos(θ) - k[2k+1] * sin(θ)
k_rot[2k+1] = k[2k]   * sin(θ) + k[2k+1] * cos(θ)
```

### Step 5: Attention Computation

```
Attention(q, k) = softmax(q_rot · k_rot^T / √d_k)
```

The dot product becomes:
```
q_rot · k_rot = q · k * cos(θ_q - θ_k)
```

Since θ ∝ position, the difference encodes relative distance.

## Why It Works

### Mathematical Insight

The key insight: cos(θ_q - θ_k) depends only on the difference in angles, which is proportional to the difference in positions.

If θ_p = p * ω and θ_q = q * ω:
```
θ_q - θ_p = (q - p) * ω ∝ (q - p)
```

So the attention weight depends on relative distance (q - p), not absolute positions.

### Geometric Interpretation

- Each dimension pair represents a 2D subspace
- Rotating vectors in this subspace by position-dependent angles
- The angle difference between two vectors encodes their positional distance
- Closer positions have smaller angle differences → higher attention
- Farther positions have larger angle differences → lower attention

## Advantages

1. **No Extra Parameters**: Same weight count as models without position encoding
2. **Relative by Construction**: Naturally encodes distance without explicit tables
3. **Smooth Extrapolation**: Angles extend indefinitely; no hard context limit
4. **Streaming-Friendly**: Rotation computed once when writing to KV cache
5. **Computationally Efficient**: Simple trigonometric operations

## NTK Scaling

### Problem

As context length increases, RoPE performance can degrade because:
- Rotation angles become very large
- Cosine of large angles oscillates rapidly
- Position information becomes ambiguous

### Solution: NTK-Aware Scaling

Adjust the base frequency to maintain performance at long contexts:
```
θ(p, k) = p * ω_k / (base_freq_scale)
```

Common approaches:
- **NTK scaling**: Dynamic adjustment based on context length
- **YaRN scaling**: Yarn-aware RoPE extension
- **Linear scaling**: Simple linear adjustment

## Implementation in Modern LLMs

### LLaMA-2/3
- Uses RoPE with NTK-aware scaling
- Supports long contexts (up to 4k/8k tokens)
- Optimized for inference

### Mistral
- Uses RoPE with custom scaling
- Efficient implementation
- Good long-context performance

### Gemma
- Uses RoPE with standard configuration
- Stable across different context lengths

## Performance Considerations

### Memory Usage
- RoPE itself adds no parameters
- KV cache still required for streaming
- Rotation computation is O(d) per token

### Computation
- Rotation: 2 multiplications + 2 additions per pair
- Total: d multiplications + d additions per token
- Can be fused with other operations for efficiency

### Inference
- Pre-compute rotation matrices for all positions
- Cache rotation angles
- Use hardware acceleration for trigonometric functions

## Optimization Opportunities for Ollama

1. **Pre-computed Rotation Tables**: Cache rotation matrices
2. **Fused Operations**: Combine RoPE with other operations
3. **Quantization**: Use lower precision for rotation angles
4. **Batch Processing**: Process multiple positions together
5. **Custom Scaling**: Optimize scaling factors for specific use cases
