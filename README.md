# NLU Sanskrit to English

IIT NLU assignment: Sanskrit → English neural machine translation using **NLLB-200-distilled-600M** fine-tuned with LoRA.

## Repository layout

```
sanskrit_en_nmt_assignment.ipynb   # Main notebook (run top-to-bottom)
submission.csv                     # Test predictions for submission
REPORT_DRAFT.md                    # Report draft (convert to PDF)
Assignment_NLU.pdf                 # Assignment brief
dl_dataset/                        # Professor-provided parallel corpus
outputs/
  metrics/                         # BLEU metrics, training curves
  checkpoints/best_v2/             # Best LoRA adapter (v2, use this)
  checkpoints/best/                # v1 LoRA adapter (earlier run)
```

## Model

- **Base:** [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Languages:** `san_Deva` → `eng_Latn`
- **Adaptation:** LoRA (r=16) on `q_proj`, `k_proj`, `v_proj`, `out_proj`
- **Decoding:** beam=5, length_penalty=0.8, no_repeat_ngram_size=3

## Results (v2)

| Split | BLEU |
|-------|------|
| Dev   | 0.290 |
| Test  | 0.286 |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets peft accelerate sentencepiece nltk bert-score pandas matplotlib tqdm
jupyter notebook sanskrit_en_nmt_assignment.ipynb
```

To regenerate `submission.csv` only, load `outputs/checkpoints/best_v2/` and run the inference cells.

## Disclosure

Pretrained NLLB-200 model from Meta/Hugging Face. Fine-tuning used only the provided `dl_dataset` files.
