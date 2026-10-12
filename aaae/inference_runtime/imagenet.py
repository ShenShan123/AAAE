"""ImageNet label lookup extracted from ai_models/src/run_sanity_checks.py."""

from __future__ import annotations


from pathlib import Path

from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


DATA_ROOT = ROOT / "dataset"


IMAGENET_GT_PATH = DATA_ROOT / "imagenet-1k" / "ILSVRC2012_devkit_t12" / "data" / "ILSVRC2012_validation_ground_truth.txt"


IMAGENET_META_PATH = DATA_ROOT / "imagenet-1k" / "ILSVRC2012_devkit_t12" / "data" / "meta.mat"


IMAGENET_SYNSET_PATH = ROOT / "src" / "resnet50" / "synset.txt"


def imagenet_gt_indices(limit: int) -> list[int]:
    meta = loadmat(str(IMAGENET_META_PATH), squeeze_me=True)
    synsets = meta["synsets"]
    id_to_wnid = {int(item["ILSVRC2012_ID"]): str(item["WNID"]) for item in synsets}
    output_wnids = [line.split()[0] for line in IMAGENET_SYNSET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    wnid_to_index = {wnid: index for index, wnid in enumerate(output_wnids)}
    raw = [int(x.strip()) for x in IMAGENET_GT_PATH.read_text(encoding="utf-8").splitlines()[:limit]]
    return [wnid_to_index[id_to_wnid[item]] for item in raw]


