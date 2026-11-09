import torch

from ad.data import MVTecDataset, denormalize, make_loader


def test_splits_disjoint_and_normal_only(fake_mvtec):
    tr = MVTecDataset("toy", "train", root=fake_mvtec, val_ratio=0.1)
    va = MVTecDataset("toy", "val", root=fake_mvtec, val_ratio=0.1)
    assert len(tr) == 18 and len(va) == 2
    assert not {s[0] for s in tr.samples} & {s[0] for s in va.samples}
    assert all(s[2] == 0 for s in tr.samples + va.samples)


def test_split_deterministic(fake_mvtec):
    a = MVTecDataset("toy", "val", root=fake_mvtec, seed=3).samples
    b = MVTecDataset("toy", "val", root=fake_mvtec, seed=3).samples
    assert a == b


def test_test_items_and_masks(fake_mvtec):
    te = MVTecDataset("toy", "test", root=fake_mvtec, resize=256, crop=224)
    assert len(te) == 5
    labels = [te[i]["label"] for i in range(len(te))]
    assert sorted(labels) == [0, 0, 0, 1, 1]
    item = next(te[i] for i in range(len(te)) if te[i]["label"] == 1)
    assert item["image"].shape == (3, 224, 224)
    assert item["mask"].shape == (1, 224, 224)
    assert set(torch.unique(item["mask"]).tolist()) == {0.0, 1.0}
    assert item["defect_type"] == "crack"


def test_loader_and_denormalize(fake_mvtec):
    dl = make_loader(MVTecDataset("toy", "train", root=fake_mvtec), batch_size=4, num_workers=0)
    b = next(iter(dl))
    assert b["image"].shape == (4, 3, 224, 224)
    rgb = denormalize(b["image"][0])
    assert rgb.shape == (224, 224, 3) and rgb.dtype.name == "uint8"
