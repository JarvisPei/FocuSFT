# FocuSFT

Official implementation of **FocuSFT**, a bilevel supervised fine-tuning method for improving long-context utilization during training.

FocuSFT adds a training-time inner loop with transient fast-weight LoRA adapters. For each batch, the adapters are reset, adapted for a few gradient steps on the same response-token objective, and then used to condition the outer SFT update. The adapters are discarded after each step and are not saved in the final checkpoint, so inference uses a standard Hugging Face causal LM with no additional overhead.

## Method

The main training step has two parts:

1. **Inner loop**: adapt fast LoRA weights on assistant response tokens using the current batch.
2. **Outer loop**: update the base model with standard response-token SFT while the adapted fast weights are active.

Both loops use the same GLM-style bidirectional-context attention mask:

- context/user/tool tokens attend bidirectionally within their turn;
- assistant tokens remain causal;
- future turns are not visible.

## Installation

```bash
git clone <repo-url>
cd FocuSFT
pip install -e .
```

Optional:

```bash
pip install -e ".[dev]"
pip install flash-attn --no-build-isolation
```

## Data Format

Training data should be parquet files with a `messages` column containing chat-style message lists:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

Optional columns:

- `tools`
- `enable_thinking`

The dataset applies the tokenizer chat template and computes loss only on assistant tokens.

## Training

Single command matching the main paper configuration:

```bash
NUM_GPUS=8 bash scripts/train_focusft_qwen25_7b.sh \
  /path/to/train.parquet \
  ./checkpoints/focusft-qwen25-7b \
  Qwen/Qwen2.5-7B
```

Equivalent direct launch:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --train_data /path/to/train.parquet \
  --output_dir ./checkpoints/focusft-qwen25-7b \
  --max_length 4096 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --seed 1234 \
  --focusft_enabled true \
  --attention_mode glm_bidir \
  --num_inner_steps 2 \
  --inner_lr 1.0 \
  --inner_grad_clip 1.0 \
  --inner_rank 32 \
  --inner_alpha 64.0 \
  --inner_layer_fraction 0.35 \
  --inner_target_modules gate_proj,up_proj,down_proj
```

The saved checkpoint is a clean Hugging Face model. Load it normally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./checkpoints/focusft-qwen25-7b")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/focusft-qwen25-7b")
```

## Reference Hyperparameters

| Parameter | Value |
| --- | --- |
| Base model | Qwen2.5-7B |
| Max length | 4096 |
| Epochs | 5 |
| Effective batch size | 32 |
| Outer LR | 1e-5 |
| Schedule | cosine, 10% warmup |
| Weight decay | 0.01 |
| Precision | BF16 |
| Inner steps | 2 |
| Inner LR | 1.0 |
| Inner grad clip | 1.0 |
| Fast LoRA rank / alpha | 32 / 64 |
| Fast LoRA target modules | `gate_proj,up_proj,down_proj` |
| Top layer fraction | 0.35 |
| Attention mode | `glm_bidir` |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is released under the MIT License.

## Repository Layout

```text
focusft/
  attention.py    # GLM-style bidirectional-context attention masks
  data.py         # chat parquet dataset and assistant-token loss masks
  memory.py       # transient fast LoRA adapters and inner loop
  models.py       # model/tokenizer loading helpers
  trainer.py      # Hugging Face Trainer integration
scripts/
  train.py
  train_focusft_qwen25_7b.sh
configs/
  focusft_qwen25_7b.yaml
tests/
```
