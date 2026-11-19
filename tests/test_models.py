import pytest
import torch

from models_mmst_vit import MMSTViT
from models_pvt_simclr import PVTSimCLR


@pytest.fixture(scope="module")
def pvt():
    torch.manual_seed(0)
    return PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9).eval()


def test_pvt_simclr_embeds_tiles_with_weather(pvt):
    x = torch.randn(2, 3, 224, 224)
    context = torch.randn(2, 28, 9)  # 28 days x 9 weather variables per tile

    with torch.no_grad():
        out = pvt(x, context=context)

    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_year", [3, 5])
def test_mmst_vit_predicts_one_vector_per_county(pvt, num_year):
    model = MMSTViT(out_dim=2, pvt_backbone=pvt, dim=512, num_year=num_year, batch_size=32).eval()
    x = torch.randn(1, 6, 4, 3, 224, 224)  # (batch, months, grids, C, H, W)
    ys = torch.randn(1, 6, 4, 28, 9)       # (batch, months, grids, days, variables)
    yl = torch.randn(1, num_year, 12, 9)   # (batch, years, months, variables)

    with torch.no_grad():
        out = model(x, ys=ys, yl=yl)

    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_mmst_vit_handles_more_grids_than_backbone_batch(pvt):
    # forward_features splits tiles into chunks of batch_size; 6 x 7 = 42 tiles -> 3 chunks of 16
    model = MMSTViT(out_dim=2, pvt_backbone=pvt, dim=512, num_year=3, batch_size=16).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 6, 7, 3, 224, 224), ys=torch.randn(1, 6, 7, 28, 9), yl=torch.randn(1, 3, 12, 9))
    assert out.shape == (1, 2)
