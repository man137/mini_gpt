# scratch_gpt/data

Training data for the from-scratch GPT lives here (git-ignored).

## Tiny Shakespeare

`train.py` downloads it automatically on first run (1.1 MB, ~1.1M characters):

```
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
mv input.txt scratch_gpt/data/
```

Manual download is useful when the training machine has no internet
(e.g. an air-gapped box or a flaky Colab session): download on your laptop
and upload the file.

## Files produced by training (also git-ignored)

- `char_tokenizer.json` - the char vocab built from the dataset
- `../checkpoints/best.pt`, `../checkpoints/last.pt` - model weights
- `../logs/train_log.csv`, `../logs/loss_curve.png` - training curves

## Optional: your own corpus

Any plain-text `.txt` works. Set the path in `common/config.py`
(`Config.dataset_path`) or edit `DEFAULT_CONFIG`. Character-level models
need at least a few MB of text to produce recognizable output; with less,
expect spelling-like gibberish that still matches the character statistics
of your file.
