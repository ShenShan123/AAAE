from __future__ import annotations

from dataclasses import dataclass


TOPOLOGY_KEY = "approx4_2_8bit_full24_v1"
DATASET_NAME = "D_mult"
N_BITS = 8
LOW_COMPRESSOR_COUNT = 9
HIGH_COMPRESSOR_COUNT = 8
VAR17_COUNT = LOW_COMPRESSOR_COUNT + HIGH_COMPRESSOR_COUNT
FULL24_COUNT = 24
TRUTH_TABLE_SHAPE = (256, 256)
M_MAX_U8_MUL = 255 * 255


NODE_KIND_TO_ID: dict[str, int] = {
    "half_adder": 0,
    "full_adder": 1,
    "exact_4_2": 2,
    "approx_4_2": 3,
    "approx_4_2_wrapper": 4,
}


COMPRESSOR_NAME_TO_ID: dict[str, int] = {
    "NONE": 0,
    "ACCI1": 1,
    "kong2": 2,
    "antonio": 3,
    "momeni": 4,
    "ha_approx": 5,
    "akbar1": 6,
    "akbar2": 7,
    "sabetz": 8,
    "E_4_2": 9,
    "Full_adder": 10,
    "Half_adder": 11,
}


COMPRESSOR_ID_TO_NAME: dict[int, str] = {value: key for key, value in COMPRESSOR_NAME_TO_ID.items()}

HIGH_DISABLED_SENTINEL = 0
EXACT_42_FAMILY_ID = COMPRESSOR_NAME_TO_ID["E_4_2"]
EXACT_42_NODE_KIND_ID = NODE_KIND_TO_ID["exact_4_2"]
APPROX_WRAPPER_NODE_KIND_ID = NODE_KIND_TO_ID["approx_4_2_wrapper"]


SPLIT_TO_CODE: dict[str, int] = {
    "train": 0,
    "test": 1,
    "val": 2,
}


@dataclass(frozen=True)
class SplitConfig:
    train: int = 72_000
    test: int = 4_000
    val: int = 4_000

    @property
    def total(self) -> int:
        return int(self.train + self.test + self.val)
