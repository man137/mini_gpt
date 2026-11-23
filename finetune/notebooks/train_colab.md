# Colab training sequence (copy-paste cells)

Free-tier T4 (16 GB VRAM) is enough for the whole pipeline. GPU runtime:
Runtime -> Change runtime type -> T4 GPU.

## Cell 1 - environment (2-3 min)

```python
!nvidia-smi
!pip install -q -U transformers==4.46.3 datasets==3.1.0 peft==0.13.2 trl==0.12.1 bitsandbytes==0.44.1 accelerate==0.34.2 sentencepiece
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Note: pin the versions above. Colab's default `transformers` moves fast and
`peft`/`trl` occasionally lag behind it; the pinned set is tested together.

## Cell 2 - get the repo in

Option A (recommended): upload the `mini-chatgpt` folder to Google Drive, or
clone your GitHub fork:

```python
# from Drive:
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/mini-chatgpt

# or from GitHub (replace USER/REPO):
# !git clone https://github.com/USER/mini-chatgpt.git
# %cd mini-chatgpt
```

## Cell 3 - prepare the data (uses the Qwen chat template)

```python
!python finetune/prepare_data.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Expected output ends with:
```
split (seed=42): train=36 val=4
wrote train/val .jsonl and _tokenized.jsonl to finetune/data
```

## Cell 4 - QLoRA training (~25-40 min on T4, approximate)

```python
!python finetune/train_qlora.py
```

Expected output (approximate numbers - record your own):
- `trainable params: ~9M || all params: ~1.5B || trainable%: ~0.6`
- loss starts around 1.5-2.0 and falls below ~1.0 within 30-50 steps
- `peak VRAM: ~6-8 GB`
- adapter saved to `finetune/outputs/adapter`

If you hit CUDA OOM, follow the numbered list in `train_qlora.py`'s docstring
(batch -> grad accumulation -> seq length -> 0.5B model -> smaller r).

## Cell 5 - merge + side-by-side comparison (~5 min)

```python
!python finetune/merge_and_test.py
```

Reads the same 5 questions with base and merged weights and prints both.
Copy interesting outputs into `eval/report_template.md` as qualitative
evidence for the interview.

## Cell 6 - download artifacts to your laptop

```python
from google.colab import files
!zip -r adapter.zip finetune/outputs/adapter
files.download('adapter.zip')          # ~20-40 MB
# merged/ is ~3 GB; leave it on Drive and serve from there, or:
# !zip -r merged.zip finetune/outputs/merged
```

## What to do if Colab disconnects

- Training state: `train_qlora.py` saves checkpoints every 50 steps
  (`save_total_limit=2`); restart and re-run cells 1-4.
- Data: prepare_data output is deterministic (seed 42) - safe to regenerate.
- Sessions idle out after ~90 min; keep the tab open during training or use
  a free background-execution toggle if your account has it.
