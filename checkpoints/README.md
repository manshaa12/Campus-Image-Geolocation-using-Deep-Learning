# Checkpoints

Large model checkpoints are intentionally not committed to this repository.

Recommended options:

1. Keep checkpoints locally under this directory while developing.
2. Upload trained weights to a GitHub Release.
3. Host trained weights on Hugging Face or another external storage service.

The training script writes metadata-rich checkpoints with this format:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "config": ...,
    "metrics": ...,
    "created_at": ...,
}
```

The evaluation and prediction scripts also support older plain `state_dict` checkpoints.
