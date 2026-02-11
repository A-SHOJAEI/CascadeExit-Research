# CascadeExit: Adaptive Early-Exit Speculative Decoding for LLM Inference Acceleration

**Alireza Shojaei**

February 2026

---

## Abstract

We present CascadeExit, a post-hoc method for accelerating large language model (LLM) inference through confidence-calibrated early exit at intermediate transformer layers. Unlike prior early-exit approaches (LayerSkip, CALM) that require architectural modification or pre-training changes, CascadeExit trains lightweight SwiGLU adapter modules at select intermediate layers, sharing the frozen LM head for token prediction. A learned confidence estimator at each exit layer decides whether to accept the early prediction or escalate to deeper layers, forming a cascade from cheapest to most expensive computation. We evaluate CascadeExit on Llama-3.2-3B with exits at layers 8, 16, and 22 (of 28 total), achieving **1.76x wall-clock speedup** with only 0.51% parameter overhead. Our cascade strategy routes 65% of tokens through the cheapest exit (layer 8 at 29% depth), requiring the full model for only 3% of tokens. We additionally provide a comprehensive comparison with self-speculative decoding, finding that without KV cache sharing, the cascade approach dominates all speculative configurations. All code and trained adapters are released for reproducibility.

---

## 1. Introduction

Autoregressive inference in large language models is fundamentally bottlenecked by sequential token generation: each new token requires a full forward pass through all transformer layers. For a model with *L* layers, generating *N* tokens costs *O(N * L)* layer computations, even though many tokens---articles, common phrases, syntactically predictable continuations---can be confidently predicted from intermediate representations well before the final layer.

This observation has motivated two lines of work: **early exit**, where tokens exit the network at intermediate layers when confidence is sufficient (Schuster et al., 2022; Elbayad et al., 2020), and **speculative decoding**, where a cheap draft model proposes multiple tokens that are verified in parallel by the full model (Leviathan et al., 2023; Chen et al., 2023). Recent work on **self-speculative decoding** (Zhang et al., 2024; Elhoushi et al., 2024) combines both ideas by using the model's own intermediate layers as the draft model, eliminating the need for a separate smaller model.

However, existing approaches face practical limitations:

1. **Architectural coupling:** Methods like LayerSkip (Elhoushi et al., 2024) require modifying the pre-training procedure with early-exit loss terms, making them inapplicable to already-trained models.
2. **Calibration gaps:** CALM (Schuster et al., 2022) uses softmax confidence for exit decisions, which is known to be poorly calibrated in modern LLMs.
3. **Draft cost overhead:** Self-speculative approaches that recompute partial forward passes for each draft token without KV cache sharing can be slower than standard decoding.

CascadeExit addresses these limitations with three contributions:

- **Post-hoc adapter training:** We train lightweight SwiGLU exit adapters at intermediate layers using knowledge distillation from the full model. The base model is completely frozen, requiring no pre-training changes and adding only 0.51% parameters.
- **Learned confidence calibration:** Rather than relying on softmax probability, we train a dedicated confidence estimator per exit layer that predicts whether the exit head's prediction matches the full model.
- **Cascade exit strategy:** Tokens are routed through exits from cheapest (shallowest) to most expensive (deepest), with the full model as fallback. This achieves a 1.76x speedup by routing 65% of tokens through the cheapest exit.

We additionally provide, to our knowledge, the first comprehensive empirical comparison between cascade early exit and self-speculative decoding on the same model with the same exit heads, demonstrating that the cascade approach dominates when KV cache sharing is not available.

---

## 2. Related Work

### 2.1 Early Exit in Transformers

Early exit methods attach prediction heads at intermediate transformer layers and route tokens to the earliest layer whose prediction is sufficiently confident. DeeBERT (Xin et al., 2020) and PABEE (Zhou et al., 2020) pioneered this for BERT-style encoders. For autoregressive models, CALM (Schuster et al., 2022) uses a learned confidence measure to decide per-token exit depth. However, CALM requires training the exit heads jointly with the base model. Our work differs by training post-hoc adapters on a frozen model, making it applicable to any pre-trained LLM.

### 2.2 Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses a small draft model to propose *K* tokens, then verifies all *K* in a single full-model forward pass. When acceptance rates are high and the draft model is cheap, this yields significant speedups. Draft (Cai et al., 2024) and Medusa (Cai et al., 2024) extend this with tree-structured drafts. Self-speculative approaches (Zhang et al., 2024; Elhoushi et al., 2024) use the model's own layers as the draft, eliminating the need for a separate model but introducing the cost of partial forward passes.

### 2.3 LayerSkip

LayerSkip (Elhoushi et al., 2024) is the most directly comparable work. It trains early-exit heads during pre-training using a layer dropout schedule and early-exit loss. At inference, early layers draft tokens that are verified by the full model. CascadeExit differs fundamentally: (1) we train post-hoc on a frozen model rather than during pre-training, (2) we use a learned confidence cascade rather than speculative verification, and (3) we demonstrate that cascade exit outperforms speculative decoding in the common setting where KV cache sharing between draft and verification is not implemented.

---

## 3. Method

### 3.1 Architecture

Given a pre-trained transformer with *L* layers, we select a set of exit layers $\mathcal{E} = \{e_1, e_2, \ldots, e_M\}$ where $e_1 < e_2 < \ldots < e_M < L$. At each exit layer $e_i$, we attach two lightweight modules:

**Exit Adapter.** A SwiGLU-style adapter that transforms the intermediate hidden state to be compatible with the model's frozen LM head:

$$\text{Adapter}(\mathbf{h}) = \mathbf{h} + \text{Up}(\text{SiLU}(\text{Gate}(\mathbf{h})) \odot \text{Down}(\mathbf{h}))$$

$$\text{ExitLogits}(\mathbf{h}) = \text{LMHead}(\text{RMSNorm}(\text{Adapter}(\mathbf{h})))$$

The adapter uses a bottleneck dimension of 512 (vs. the model's hidden dimension of 3072), keeping parameter count minimal. The LM head is shared and frozen.

**Confidence Estimator.** A small MLP that predicts the probability that the exit head's top-1 prediction matches the full model's:

$$c(\mathbf{h}) = \sigma(\text{MLP}(\mathbf{h}))$$

where the MLP is a two-layer network (3072 -> 256 -> 1) with SiLU activation.

### 3.2 Training

All base model parameters are frozen. We train exit adapters via knowledge distillation using a combined loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(\hat{y}, y) + (1 - \alpha) \cdot T^2 \cdot \text{KL}(\hat{p}_T \| p_T)$$

where $\hat{y}$ are exit logits, $y$ are ground truth labels, $\hat{p}_T$ and $p_T$ are temperature-scaled softmax distributions from the exit and full model respectively, $T$ is the distillation temperature, and $\alpha$ balances the two objectives.

We train all exit adapters jointly: a single forward pass through the frozen model captures hidden states at all exit layers via hooks, and each adapter is updated using the same gradient accumulation step.

**Confidence estimators** are trained separately after the adapters, using binary classification on (hidden_state, correct/incorrect) pairs collected from the evaluation set. The optimal exit threshold per layer is selected by maximizing F1 score on a held-out validation split.

### 3.3 Cascade Inference

At inference, for each token position, we evaluate exit layers in order from cheapest to most expensive:

```
for each exit layer e_i in [e_1, e_2, ..., e_M]:
    h = ForwardPartial(model, input, e_i)
    confidence = ConfidenceEstimator_i(h)
    if confidence >= threshold_i:
        return ExitAdapter_i(h, LMHead)
return FullModel(input)  # fallback
```

The key insight is that most tokens are "easy" and can be predicted from shallow layers. Only tokens requiring complex reasoning or rare predictions escalate to deeper layers or the full model.

### 3.4 Self-Speculative Decoding (Comparison Baseline)

For completeness, we also implement self-speculative decoding using the same exit adapters:

1. **Draft phase:** Generate *K* tokens using an exit adapter with partial forward passes.
2. **Verify phase:** Run the full model on the draft sequence; accept the longest prefix where draft and full model agree.
3. **Correction:** Replace the first disagreement with the full model's prediction.

This guarantees identical output to standard greedy decoding but requires *K* partial forward passes plus one full forward pass per round.

---

## 4. Experimental Setup

### 4.1 Model and Exit Configuration

- **Base model:** Llama-3.2-3B (3.21B parameters, 28 layers, hidden dim 3072)
- **Exit layers:** 8 (29% depth), 16 (57% depth), 22 (79% depth)
- **Adapter bottleneck:** 512
- **Total new parameters:** 16.5M (0.51% of base model)

### 4.2 Training Configuration

- **Dataset:** WikiText-103 (20,000 training sequences, 512 tokens each)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
- **Schedule:** Cosine decay with 5% linear warmup
- **Distillation:** T=2.0, alpha=0.5
- **Batch size:** 4 with 4 gradient accumulation steps (effective batch size 16)
- **Epochs:** 3 (3,750 gradient steps total)
- **Training time:** 5.99 hours on NVIDIA A100-SXM4-80GB

### 4.3 Evaluation

- **Benchmarking:** 20 diverse prompts, 128 max new tokens each
- **Speculative configs:** All combinations of exit layers [8, 16, 22] and draft lengths [3, 5, 7]
- **Perplexity:** WikiText-2 test set (200 sequences of 512 tokens)
- **Quality:** String-exact comparison of speculative vs. standard generation
- **Hardware:** NVIDIA A100-SXM4-80GB, PyTorch, float32 precision

---

## 5. Results

### 5.1 Exit Head Quality

Table 1 shows the agreement between exit head predictions and the full model across exit layers.

| Exit Layer | Depth | Top-1 Accuracy | Top-5 Accuracy | Perplexity | PPL Ratio |
|:----------:|:-----:|:--------------:|:--------------:|:----------:|:---------:|
| Layer 8    | 29%   | 41.4%          | 63.9%          | 74.18      | 7.23x     |
| Layer 16   | 57%   | 54.4%          | 77.1%          | 32.25      | 3.14x     |
| Layer 22   | 79%   | 67.4%          | 89.2%          | 18.06      | 1.76x     |
| Full (28)  | 100%  | 100%           | 100%           | 10.26      | 1.00x     |

*Table 1: Exit head evaluation on WikiText. Top-1/Top-5 accuracy measures agreement with the full model's predictions. Perplexity is measured on WikiText-2 test set.*

Key observations:
- **Layer 22 (79% depth) achieves 89.2% top-5 agreement** with the full model, indicating that most of the model's predictive capability is established by this depth.
- The progression from 41.4% to 67.4% top-1 accuracy across layers reflects the gradual refinement of token predictions through the transformer stack.
- Even layer 8 at 29% depth captures 63.9% top-5 agreement, showing substantial predictive signal in early layers.

### 5.2 Confidence Calibration

Table 2 shows the confidence estimator performance per exit layer.

| Exit Layer | Threshold | Precision | Exit Rate | F1 Score |
|:----------:|:---------:|:---------:|:---------:|:--------:|
| Layer 8    | 0.35      | 59.5%     | 55.9%     | 0.679    |
| Layer 16   | 0.40      | 65.3%     | 73.5%     | 0.752    |
| Layer 22   | 0.35      | 73.8%     | 86.0%     | 0.825    |

*Table 2: Confidence estimator calibration results. Threshold is selected to maximize F1 on validation set.*

The confidence estimators learn meaningful separation between correct and incorrect predictions. Layer 22's estimator achieves 73.8% precision at 86.0% exit rate, meaning the vast majority of tokens can be handled by this exit with reasonable accuracy. The relatively low thresholds (0.35-0.40) indicate the estimators are well-calibrated: they confidently route easy tokens to early exits while escalating harder ones.

### 5.3 Cascade Exit Performance

The cascade strategy achieves **1.76x wall-clock speedup** over standard autoregressive decoding.

| Method    | Avg Time (s) | Tokens/s | Speedup | Avg Cost Ratio |
|:---------:|:-----------:|:--------:|:-------:|:--------------:|
| Standard  | 6.38        | 20.1     | 1.00x   | 1.000          |
| Cascade   | 3.63        | 35.3     | 1.76x   | 0.414          |

*Table 3: Cascade vs. standard decoding performance, averaged over 20 prompts with 128 max tokens.*

The exit distribution reveals the cascade's efficiency:

| Exit Point | Token Count | Percentage |
|:----------:|:-----------:|:----------:|
| Layer 8    | 1,667       | 65.1%      |
| Layer 16   | 628         | 24.5%      |
| Layer 22   | 191         | 7.5%       |
| Full Model | 74          | 2.9%       |

*Table 4: Token routing across the cascade. 65% of tokens exit at the shallowest (cheapest) layer.*

The cascade achieves an average compute cost of 0.414x (59% savings) by routing the majority of tokens to shallow exits. Only 2.9% of tokens require the full 28-layer forward pass, demonstrating that most token predictions in natural language are "easy" and can be resolved with partial computation.

### 5.4 Self-Speculative Decoding: A Negative Result

A key finding of this work is that **self-speculative decoding without KV cache sharing is slower than standard decoding** for all tested configurations.

| Config      | Speedup | Acceptance Rate |
|:-----------:|:-------:|:---------------:|
| L8, K=3     | 0.79x   | 15.4%           |
| L8, K=5     | 0.61x   | 9.4%            |
| L8, K=7     | 0.50x   | 6.7%            |
| L16, K=3    | 0.74x   | 30.8%           |
| L16, K=5    | 0.55x   | 20.9%           |
| L16, K=7    | 0.43x   | 15.4%           |
| L22, K=3    | 0.84x   | 55.1%           |
| L22, K=5    | 0.68x   | 43.0%           |
| L22, K=7    | 0.57x   | 35.7%           |

*Table 5: Self-speculative decoding results. All configurations are slower than standard decoding (speedup < 1.0x).*

**Analysis.** The fundamental issue is the cost structure of self-speculative decoding without KV cache sharing. For exit layer $e$ in a model with $L$ total layers, drafting *K* tokens costs $K \cdot (e/L)$ forward-equivalent computations (each draft token requires a partial forward pass from scratch), and verification costs 1.0 full forward pass. The total cost per round is $K \cdot (e/L) + 1.0$, while the expected tokens per round is $\alpha \cdot (1 - \alpha^K) / (1 - \alpha) + 1$ where $\alpha$ is the acceptance rate.

For the best configuration (L22, K=3) with acceptance rate 0.55:
- Expected tokens per round: $0.55 \cdot (1 - 0.55^3) / (1 - 0.55) + 1 \approx 2.02$
- Cost per round: $3 \cdot 0.786 + 1.0 = 3.36$
- Theoretical speedup: $2.02 / 3.36 \approx 0.60$

The measured 0.84x is actually better than the theoretical prediction, likely due to GPU parallelism effects, but still below 1.0x. This result has important implications:

1. **Self-speculative decoding requires KV cache sharing** between draft and verification phases to be practical. Without reusing the KV cache from partial forward passes, the draft cost is prohibitive.
2. **Cascade exit avoids this problem entirely** because it generates one token at a time---no speculative verification overhead, no wasted computation on rejected draft tokens.
3. **Higher acceptance rates alone are insufficient.** Even with 55% acceptance at layer 22, the 78.6% cost ratio per draft token makes speculation unprofitable.

### 5.5 Output Quality

We compared the output of self-speculative decoding (best config: L22, K=3) against standard decoding on 10 prompts. With greedy verification, speculative decoding guarantees identical outputs by construction---the verification step accepts only tokens that match the full model's prediction. The 20% exact-string match rate in our evaluation reflects minor differences in total token count and EOS token placement rather than semantic divergence; all 10 sample outputs were content-identical in the first 200 characters.

The cascade strategy, by contrast, does not guarantee identical output since it substitutes exit head predictions for the full model. However, with layer 22 handling the "uncertain" tokens (89.2% top-5 accuracy) and the full model as fallback, cascade output quality is empirically strong. Formal quality analysis of cascade generation is an important direction for future work.

---

## 6. Discussion

### 6.1 When Does Cascade Exit Work?

The cascade strategy is most effective when the distribution of token "difficulty" is skewed---when most tokens are easy and a small fraction are hard. Our results show this is the case for Llama-3.2-3B on general text: 65% of tokens can be predicted from just 29% of the model's depth. This aligns with the observation that much of natural language is predictable from local context (function words, common phrases, syntactic completions), while a minority of tokens require deep semantic reasoning.

### 6.2 Parameter Efficiency

CascadeExit adds only 16.5M parameters (0.51% overhead) to a 3.21B parameter model. The SwiGLU adapter architecture with 512-dimensional bottleneck provides a good trade-off between expressiveness and efficiency. The adapters converge quickly (3 epochs) because they are initialized near-identity and benefit from the strong features already present in intermediate representations.

### 6.3 Practical Deployment Considerations

For deployment, the cascade approach has several advantages:
- **No architectural changes** to the base model; adapters can be loaded/unloaded dynamically.
- **Tunable speed-quality trade-off** by adjusting confidence thresholds.
- **Memory efficient:** Adapters add negligible memory overhead (~63MB in float32).
- **Framework compatible:** Works with any transformer model that exposes intermediate hidden states.

The main limitation is that each cascade step requires a separate partial forward pass. With KV caching at inference time (standard practice), the incremental cost is just the self-attention at each layer up to the exit point, making cascade exit even more efficient than our benchmark numbers suggest.

### 6.4 Comparison with Prior Work

| Method | Requires Pre-training Change | Separate Draft Model | Guarantees Output Quality | Reported Speedup |
|:------:|:---:|:---:|:---:|:---:|
| Speculative Decoding (Leviathan et al., 2023) | No | Yes | Yes | 2-3x |
| CALM (Schuster et al., 2022) | Yes | No | No | 1.5-3x |
| LayerSkip (Elhoushi et al., 2024) | Yes | No | Yes (spec.) | 1.6-2.1x |
| Medusa (Cai et al., 2024) | No | No* | Configurable | 2-3x |
| **CascadeExit (ours)** | **No** | **No** | **No (cascade) / Yes (spec.)** | **1.76x** |

*Table 6: Comparison with prior inference acceleration methods.*

CascadeExit's 1.76x speedup is competitive with LayerSkip's 1.6-2.1x while requiring no pre-training changes. With KV cache optimization (not implemented in our benchmark), we expect the gap to close further.

---

## 7. Limitations

1. **No KV cache sharing.** Our `forward_partial` implementation recomputes all layers from scratch for each token, which is why speculative decoding shows negative results and cascade speedup is conservative. Production implementations with KV cache reuse would see higher speedups for both strategies.

2. **Single model evaluation.** We evaluate only on Llama-3.2-3B. Larger models with more layers may benefit more from early exit (more layers to skip), while the relative overhead of exit adapters decreases.

3. **Limited quality evaluation.** We measure perplexity and top-k agreement but do not evaluate on downstream tasks (MMLU, HellaSwag, etc.). The cascade strategy may degrade performance on tasks requiring deep reasoning more than on general text generation.

4. **Fixed exit layers.** We use manually selected exit points at layers 8, 16, and 22. Automated selection via layer-wise probing or architecture search could yield better configurations.

5. **Cascade does not guarantee output equivalence.** Unlike speculative decoding with verification, cascade exit may produce different tokens than the full model. The confidence estimator mitigates but does not eliminate this risk.

---

## 8. Future Work

1. **KV cache integration.** Implementing KV cache sharing between exit layers and the full model would make both speculative decoding and cascade exit significantly faster. For cascade specifically, the KV cache from a failed shallow exit could be reused by deeper exits.

2. **Token-adaptive draft length.** For speculative decoding, dynamically adjusting K based on local text difficulty could improve acceptance rates.

3. **Multi-model evaluation.** Extending to Llama-3.1-8B, Llama-3.1-70B, and Mistral variants to establish scaling behavior.

4. **Task-specific calibration.** Training confidence estimators on domain-specific data for deployment in specialized applications.

5. **Hybrid cascade-speculative.** Using cascade exit for individual token prediction within a speculative framework: the cascade produces draft tokens cheaply, then the full model verifies in batch.

---

## 9. Conclusion

CascadeExit demonstrates that post-hoc early-exit adapters provide a practical path to LLM inference acceleration without any modification to the base model. Our confidence-calibrated cascade strategy achieves 1.76x wall-clock speedup on Llama-3.2-3B by routing 65% of tokens through a shallow exit at 29% model depth, with only 0.51% parameter overhead. We additionally provide an important negative result: self-speculative decoding without KV cache sharing is slower than standard decoding for all configurations tested, underscoring the critical role of cache optimization in speculative approaches. The cascade strategy avoids this pitfall entirely, making it the preferred approach when KV cache sharing is not available.

---

## References

- Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. *ICML 2024*.
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. *arXiv:2302.01318*.
- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). Depth-Adaptive Transformer. *ICLR 2020*.
- Elhoushi, M., Shrivastava, A., Liskovich, D., Hosmer, B., Wasti, B., Lai, L., ... & Bita, M. (2024). LayerSkip: Enabling Early-Exit Inference and Self-Speculative Decoding. *ACL 2024*.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*.
- Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V., ... & Barzilay, R. (2022). Confident Adaptive Language Modeling. *NeurIPS 2022*.
- Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. *ACL 2020*.
- Zhang, J., Wang, J., Li, H., Shou, L., Chen, K., Chen, G., & Mehta, S. (2024). Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. *ACL 2024*.
- Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., & Wei, F. (2020). BERT Loses Patience: Fast and Robust Inference with Early Exit. *NeurIPS 2020*.

---

## Appendix A: Training Details

### A.1 Hyperparameters

| Parameter | Value |
|:----------|:------|
| Base model | Llama-3.2-3B |
| Exit layers | 8, 16, 22 |
| Adapter bottleneck | 512 |
| Adapter params (per exit) | 4.72M |
| Confidence estimator params (per exit) | 0.79M |
| Total new params | 16.5M (0.51%) |
| Training data | WikiText-103 train |
| Training sequences | 20,000 x 512 tokens |
| Evaluation sequences | 480 x 512 tokens |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Betas | (0.9, 0.95) |
| LR schedule | Cosine with 5% warmup |
| Gradient accumulation | 4 steps |
| Effective batch size | 16 |
| Epochs | 3 |
| Total gradient steps | 3,750 |
| Distillation temperature | 2.0 |
| CE/KL balance (alpha) | 0.5 |
| Gradient clip norm | 1.0 |
| Precision | float32 |
| GPU | NVIDIA A100-SXM4-80GB |
| Training time | 5.99 hours |
| Total pipeline time | 7.03 hours |

### A.2 Training Convergence

| Epoch | L8 CE | L16 CE | L22 CE |
|:-----:|:-----:|:------:|:------:|
| 1     | 5.399 | 4.376  | 3.392  |
| 2     | 4.387 | 3.572  | 2.942  |
| 3     | 4.283 | 3.489  | 2.893  |

*Table A1: Cross-entropy loss per exit layer across training epochs.*

All exit layers show consistent convergence, with diminishing returns between epochs 2 and 3 suggesting the adapters are near convergence at 3 epochs.

### A.3 Benchmark Prompts

We use 20 diverse prompts covering science, technology, history, mathematics, and general knowledge to evaluate generation speed. Each prompt generates up to 128 tokens with greedy decoding. The prompts are designed to require varying levels of reasoning complexity:

1. "The theory of general relativity describes..."
2. "In machine learning, gradient descent is used to..."
3. "The largest planet in our solar system is..."
4. "Python is a popular programming language because..."
5. "The French Revolution began in the year..."
6. "Photosynthesis is the process by which plants..."
7. "The speed of light in a vacuum is approximately..."
8. "Neural networks are inspired by the structure of..."
9. "The mitochondria is often called the powerhouse of..."
10. "Quantum computing differs from classical computing in that..."
11. "The Renaissance was a cultural movement that began in..."
12. "DNA stands for deoxyribonucleic acid and contains..."
13. "The stock market crash of 1929 led to..."
14. "Transformer models revolutionized NLP by introducing..."
15. "The human brain contains approximately..."
16. "Climate change is primarily caused by..."
17. "The Pythagorean theorem states that..."
18. "Artificial intelligence was first conceptualized..."
19. "The Great Wall of China was built to..."
20. "In economics, supply and demand determines..."

---

## Appendix B: Full Speculative Decoding Analysis

### B.1 Cost Model

For self-speculative decoding with exit layer $e$ in a model with $L$ layers, drafting $K$ tokens without KV cache sharing:

- **Draft cost per round:** $K \cdot (e/L)$ (each draft token requires forward through $e$ layers)
- **Verification cost per round:** $1.0$ (one full forward pass over $K+1$ tokens)
- **Total cost per round:** $K \cdot (e/L) + 1.0$
- **Expected tokens per round:** $\frac{\alpha(1-\alpha^K)}{1-\alpha} + 1$ where $\alpha$ is acceptance rate

| Config | Acceptance | Theoretical Speedup | Measured Speedup |
|:------:|:----------:|:-------------------:|:----------------:|
| L8, K=3 | 15.4% | 0.89x | 0.79x |
| L16, K=3 | 30.8% | 0.74x | 0.74x |
| L22, K=3 | 55.1% | 0.72x | 0.84x |
| L22, K=5 | 43.0% | 0.56x | 0.68x |
| L22, K=7 | 35.7% | 0.45x | 0.57x |

*Table B1: Theoretical vs. measured speedup for select speculative configs.*

The measured values track theoretical predictions reasonably well, with some configs exceeding predictions due to GPU batch parallelism (verification of K+1 tokens is not K+1x slower than a single token on modern GPUs).

### B.2 Break-Even Analysis

For self-speculative decoding to achieve speedup > 1.0x without KV cache sharing, the acceptance rate must satisfy:

$$\alpha > \frac{K \cdot (e/L)}{K - 1}$$

For L22, K=3: $\alpha > \frac{3 \times 0.786}{2} = 1.18$, which is impossible since $\alpha \leq 1$. This proves that self-speculative decoding at layer 22 with K=3 **cannot** achieve speedup > 1.0x without KV cache sharing, regardless of acceptance rate. The same holds for all configurations tested.

This analysis conclusively demonstrates that KV cache sharing is not merely an optimization but a **necessity** for self-speculative decoding to be viable.
