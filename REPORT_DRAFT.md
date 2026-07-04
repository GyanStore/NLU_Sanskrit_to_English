# Sanskrit→English NMT — Report Draft (v2)

## 1. Introduction

Fine-tuned **facebook/nllb-200-distilled-600M** with LoRA on 10,000 provided Sanskrit–English pairs (`san_Deva` → `eng_Latn`). No external data or APIs.

## 2. Architecture

| Component | Choice |
|-----------|--------|
| Base model | NLLB-200-distilled-600M (Meta) |
| Adaptation | LoRA r=16, α=32 on `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Trainable params | 4,718,592 (0.76%) |
| Decoding | Beam=5, length_penalty=0.8, no_repeat_ngram_size=3 |

## 3. Training

- NFC text normalization; max length 128
- AdamW lr=2e-4, weight decay=0.01, warmup 6%
- Batch 2 × grad accum 8; **6 epochs** with early stopping on dev loss
- Apple M4 Max (MPS), LoRA full-precision

## 4. Results

| Metric | Dev | Test |
|--------|-----|------|
| BLEU (NLTK, method1) | **0.2903** | **0.2858** |
| Inference time | ~6 min / 1000 | ~6 min / 1000 |

Checkpoint: `outputs/checkpoints/best_v2/`

## 5. Disclosure

Pretrained **facebook/nllb-200-distilled-600M**. Fine-tuning on provided train split only.

## 6. Discussion

Initial run (4 epochs, LoRA on q/v only, default decoding) underperformed on BLEU due to narrower adapters and weaker beam-search settings—not a wrong model choice. v2 matched best-practice NLLB+LoRA setup and exceeded prior baseline.

## 7. References

NLLB (2022); LoRA (2021); BLEU (2002); BERTScore (2020); HuggingFace Transformers.
