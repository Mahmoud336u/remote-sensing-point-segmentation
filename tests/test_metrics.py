import numpy as np
import pytest
import torch
from src.evaluate import confusion_update, metrics_from_confusion


def test_global_metrics_background_absent_fp_only():
    cm = np.array([[8, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    metrics = metrics_from_confusion(cm)
    assert metrics["per_class_iou"] == [8 / 11, 4 / 8, 0., None, None]
    assert metrics["miou"] == pytest.approx((8 / 11 + .5 + 0) / 3)
    assert metrics["pixel_accuracy"] == 12 / 16
    assert metrics["class_support"] == [10, 6, 0, 0, 0]


def test_confusion_orientation_padding_and_global_pooling():
    cm = torch.zeros(3, 3, dtype=torch.int64)
    target = torch.tensor([0, 0, 1, 2, 255])
    prediction = torch.tensor([0, 1, 2, 2, 1])
    valid = torch.tensor([True, True, True, True, False])
    confusion_update(cm, prediction[:2], target[:2], valid[:2])
    confusion_update(cm, prediction[2:], target[2:], valid[2:])
    assert cm.tolist() == [[1, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert metrics_from_confusion(cm)["valid_pixels"] == 4


def test_empty_and_perfect():
    assert metrics_from_confusion(np.zeros((5, 5), int))["miou"] is None
    m = metrics_from_confusion(np.eye(5, dtype=int))
    assert m["miou"] == m["mean_dice"] == m["pixel_accuracy"] == 1.


def test_bad_prediction():
    with pytest.raises(ValueError):
        confusion_update(torch.zeros(2, 2, dtype=torch.int64), torch.tensor([2]),
                         torch.tensor([1]), torch.tensor([True]))
