# CascadeExit: Adaptive Early-Exit Speculative Decoding for LLM Inference Acceleration

Post-hoc method for accelerating LLM inference through confidence-calibrated early exit at intermediate transformer layers. Trains lightweight SwiGLU adapter modules at select layers using knowledge distillation, with learned confidence estimators forming a cascade routing strategy from cheapest to most expensive computation.

## Key Results

| Configuration | Speedup | Parameter Overhead |
|--------------|---------|-------------------|
| CascadeExit (L8/L16/L22) | **1.76x** | 0.51% (16.5M params) |
| Best Speculative (L22, K=3) | 0.84x | Same adapters |
| Standard Decoding | 1.00x | Baseline |

### Token Routing Distribution (Cascade)

| Exit Layer | Depth | Tokens Routed | Top-1 Accuracy | Perplexity |
|-----------|-------|--------------|----------------|------------|
| Layer 8 | 29% | 65.1% | 41.4% | 74.18 |
| Layer 16 | 57% | 24.5% | 54.4% | 32.25 |
| Layer 22 | 79% | 7.5% | 67.4% | 18.06 |
| Full Model | 100% | 2.9% | 100% | 10.26 |

## Approach

1. **Post-hoc adapter training**: SwiGLU exit adapters at layers 8, 16, 22 of Llama-3.2-3B, trained via knowledge distillation from the full model (frozen base, 3 epochs on WikiText-103)
2. **Learned confidence calibration**: Dedicated binary confidence estimator per exit layer predicting whether the early prediction matches the full model
3. **Cascade exit strategy**: Tokens route from shallowest to deepest exit, with the full model as fallback. Average compute cost: 0.414x of full forward pass

Finding: Self-speculative decoding without KV cache sharing is slower than standard decoding (0.5-0.84x), making the cascade approach strictly dominant under this constraint.

## Project Structure

```
CascadeExit-Research/
├── CascadeExit_Speculative_Decoding.ipynb  # Full research pipeline
├── CascadeExit_Paper.md                    # Research paper
├── checkpoints/                            # Trained adapters + confidence estimators
├── results/                                # Evaluation metrics (JSON)
├── logs/                                   # Training logs
└── FINAL_SUMMARY.json                      # Complete execution summary
```

## Usage

The full pipeline is in the Jupyter notebook. Requires an NVIDIA GPU with sufficient VRAM and HuggingFace access to Llama-3.2-3B.

```bash
pip install torch transformers accelerate datasets
jupyter notebook CascadeExit_Speculative_Decoding.ipynb
```

**Hardware**: Developed and tested on NVIDIA A100-SXM4-80GB. Total compute: 7.03 hours.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
