import pytest
import torch
from torch.nn import functional as F

from src.losses import PartialCrossEntropyLoss


@pytest.mark.parametrize("batch,classes", [(1, 2), (3, 5), (2, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_full_equivalence(batch, classes, dtype):
    logits = torch.randn(batch, classes, 5, 7, dtype=dtype, requires_grad=True)
    target = torch.randint(classes, (batch, 5, 7))
    result = PartialCrossEntropyLoss()(logits, target, torch.ones_like(target, dtype=torch.bool))
    reference = F.cross_entropy(logits.double() if dtype == torch.float64 else logits.float(), target)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)


def test_batch_count_weighted():
    logits = torch.tensor([[[[2., 3.]], [[0., 0.]]], [[[0., 0.]], [[2., 3.]]]],
                          requires_grad=True)
    target = torch.zeros(2, 1, 2, dtype=torch.long)
    mask = torch.tensor([[[1, 0]], [[1, 1]]], dtype=torch.bool)
    reference = F.cross_entropy(logits.permute(0, 2, 3, 1)[mask], target[mask])
    torch.testing.assert_close(PartialCrossEntropyLoss()(logits, target, mask), reference)


@pytest.mark.parametrize("invalid", [-1, 5, 255])
def test_invalid_labels_and_unselected_nan(invalid):
    logits = torch.tensor([[[[1., float("nan"), float("nan")]],
                            [[2., float("nan"), float("nan")]],
                            [[3., float("nan"), float("nan")]],
                            [[4., float("nan"), float("nan")]],
                            [[5., float("nan"), float("nan")]]]], requires_grad=True)
    target = torch.tensor([[[2, 1, invalid]]])
    mask = torch.tensor([[[1, 0, 1]]], dtype=torch.bool)
    loss = PartialCrossEntropyLoss()(logits, target, mask)
    expected = F.cross_entropy(logits[:, :, :, 0], target[:, :, 0])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[..., 1:]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_empty_autograd_zero_even_with_nan(dtype):
    logits = torch.full((2, 5, 3, 3), float("nan"), dtype=dtype, requires_grad=True)
    targets = torch.full((2, 3, 3), 255, dtype=torch.long)
    loss = PartialCrossEntropyLoss()(logits, targets, torch.zeros_like(targets, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_selected_gradients_only():
    logits = torch.randn(2, 5, 3, 4, dtype=torch.double, requires_grad=True)
    target = torch.randint(5, (2, 3, 4))
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 1, 2] = mask[1, 2, 3] = True
    loss = PartialCrossEntropyLoss()(logits, target, mask)
    loss.backward()
    expected = torch.zeros_like(logits)
    selected = logits.detach().permute(0, 2, 3, 1)[mask].softmax(1)
    selected[torch.arange(2), target[mask]] -= 1
    expected.permute(0, 2, 3, 1)[mask] = selected / 2
    torch.testing.assert_close(logits.grad, expected)


def test_input_contracts():
    logits = torch.randn(1, 5, 3, 4)
    targets = torch.zeros(1, 3, 4, dtype=torch.long)
    with pytest.raises(TypeError):
        PartialCrossEntropyLoss()(logits, targets.float(), targets.bool())
    with pytest.raises(ValueError):
        PartialCrossEntropyLoss()(logits, targets, torch.full_like(targets, 2))
    with pytest.raises(ValueError):
        PartialCrossEntropyLoss()(logits[:, :, :2], targets, targets.bool())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_loss():
    x = torch.randn(2, 5, 8, 8, device="cuda", requires_grad=True)
    y = torch.randint(5, (2, 8, 8), device="cuda")
    loss = PartialCrossEntropyLoss()(x, y, torch.ones_like(y, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
