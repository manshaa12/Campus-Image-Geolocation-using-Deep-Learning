import torch

from img2gps.model import GPSGridModel


def test_decode_soft_topk_shape():
    logits = torch.zeros(4, 100)
    preds = GPSGridModel.decode_soft_topk(logits, top_k=5)
    assert preds.shape == (4, 2)


def test_decode_invalid_topk():
    logits = torch.zeros(1, 100)
    try:
        GPSGridModel.decode_soft_topk(logits, top_k=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
