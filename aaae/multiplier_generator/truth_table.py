"""Core implementation extracted from approx_error_model/model/dmult_neural/train_dmult_truth_table.py."""

from __future__ import annotations


import numpy as np

import torch

import torch.nn.functional as F

from torch import nn

from aaae.multiplier_generator.data import DMULT_TARGET_COLUMNS

TABLE_SIDE = 256


UINT16_MAX = int(np.iinfo(np.uint16).max)


M_MAX_U8_MUL = 255 * 255


def exact_table_np() -> np.ndarray:
    values = np.arange(TABLE_SIDE, dtype=np.int32)
    exact = values[:, None] * values[None, :]
    exact[0, :] = 0
    exact[:, 0] = 0
    return exact.astype(np.int32, copy=False)


def exact_table_torch(device: torch.device) -> torch.Tensor:
    values = torch.arange(TABLE_SIDE, dtype=torch.float32, device=device)
    exact = values[:, None] * values[None, :]
    exact[0, :] = 0.0
    exact[:, 0] = 0.0
    return exact


def operand_bit_tables_np() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_values = np.repeat(np.arange(TABLE_SIDE, dtype=np.uint16), TABLE_SIDE)
    col_values = np.tile(np.arange(TABLE_SIDE, dtype=np.uint16), TABLE_SIDE)
    a_bits = np.stack([((row_values >> bit) & 1).astype(np.float32) for bit in range(8)], axis=0)
    b_bits = np.stack([((col_values >> bit) & 1).astype(np.float32) for bit in range(8)], axis=0)
    zero_mask = ((row_values == 0) | (col_values == 0)).astype(np.bool_)
    return a_bits, b_bits, zero_mask


def _bool_full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    total = int(a) + int(b) + int(cin)
    return total & 1, 1 if total >= 2 else 0


def _bool_approx_compressor(token: int, a: int, b: int, c: int, d: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    if int(token) == 1:
        part1 = a ^ b
        part2 = c ^ d
        part3 = a | b
        part5 = a & b
        part7 = (1 - part5) | (1 - c)
        part8 = 1 - ((1 - part2) ^ (1 - (part3 & (1 - part5))))
        s = 1 - (part7 & part8)
        carry = 1 - (((1 - part2) | (1 - (part1 & (1 - part5)))) & (1 - part5) & ((1 - c) | (1 - d)))
    elif int(token) == 2:
        part1 = a ^ b
        part2 = c ^ d
        part3 = a | b
        part4 = c | d
        part5 = a & b
        part6 = c & d
        s = (part1 & (1 - part2)) | ((1 - part1) & part4)
        carry = 1 - (((1 - part3) & (1 - part6)) | ((1 - part4) & (1 - part5)))
    elif int(token) == 3:
        w1 = a | b
        w2 = c | (a & b)
        s, carry = _bool_full_adder(w1, w2, d)
    elif int(token) == 4:
        s = (1 - (a ^ b)) | (1 - (c ^ d))
        carry = (a | b) & (c | d)
    elif int(token) == 5:
        w1 = a ^ b
        w2 = c | d
        s = w1 ^ w2
        carry = (a & b) | (w1 & w2)
    elif int(token) == 6:
        s = (a ^ b) | (c ^ d)
        carry = d
    elif int(token) == 7:
        s = (a ^ b) | (c ^ d)
        carry = (a & b) | (c & d)
    elif int(token) == 8:
        s = 1
        carry = (d & (a | c)) | (a & c)
    else:
        total = a + b + c + d
        s = total & 1
        carry = (total >> 1) & 1
    return int(s), int(carry)


def known_compressor_lut_np(num_tokens: int = 9) -> np.ndarray:
    lut = np.zeros((int(num_tokens), 16, 2), dtype=np.float32)
    for token in range(int(num_tokens)):
        for idx in range(16):
            a = (idx >> 3) & 1
            b = (idx >> 2) & 1
            c = (idx >> 1) & 1
            d = idx & 1
            lut[token, idx] = _bool_approx_compressor(token, a, b, c, d)
    return lut


class NeuralLogicTruthDecoder(nn.Module):
    """Topology-shaped neural decoder for the fixed 8-bit D_mult circuit.

    The data-dependent choices are the nine low-part compressor tokens and,
    for 17-slot checkpoints, the eight high-wrapper compressor tokens.  High
    token 0 means exact E_4_2; high tokens 1..8 select AE_4_2 wrappers around
    the same 4-input/2-output compressor LUT.  The surrounding reduction
    topology is fixed and differentiable, so this remains a neural forward pass
    and does not call the Verilog simulator used to build labels.
    """

    def __init__(
        self,
        *,
        num_slots: int = 9,
        num_tokens: int = 9,
        lut_init: str = "known",
        trainable_lut: bool = False,
        lut_logit_scale: float = 30.0,
    ) -> None:
        super().__init__()
        if int(num_slots) not in {9, 17}:
            raise ValueError("NeuralLogicTruthDecoder supports 9-slot low-only or 17-slot low+high D_mult topology.")
        self.num_slots = int(num_slots)
        self.num_tokens = int(num_tokens)
        self.trainable_lut = bool(trainable_lut)

        a_bits, b_bits, zero_mask = operand_bit_tables_np()
        self.register_buffer("a_bits", torch.from_numpy(a_bits), persistent=False)
        self.register_buffer("b_bits", torch.from_numpy(b_bits), persistent=False)
        self.register_buffer("zero_operand_mask_flat", torch.from_numpy(zero_mask), persistent=False)
        self.register_buffer("exact_flat", torch.from_numpy(exact_table_np().reshape(-1).astype(np.float32)), persistent=False)

        if str(lut_init).strip().lower() == "known":
            lut = torch.from_numpy(known_compressor_lut_np(num_tokens=int(num_tokens)))
        elif str(lut_init).strip().lower() == "random":
            lut = torch.rand(int(num_tokens), 16, 2, dtype=torch.float32)
            lut[0] = torch.from_numpy(known_compressor_lut_np(num_tokens=int(num_tokens))[0])
        else:
            raise ValueError(f"Unsupported lut_init={lut_init!r}; expected known or random.")

        if bool(trainable_lut):
            if str(lut_init).strip().lower() == "known":
                logits = ((lut * 2.0) - 1.0) * float(lut_logit_scale)
            else:
                eps = 1.0e-4
                logits = torch.logit(torch.clamp(lut, eps, 1.0 - eps))
                logits[0] = ((torch.from_numpy(known_compressor_lut_np(num_tokens=int(num_tokens))[0]) * 2.0) - 1.0) * float(lut_logit_scale)
            self.compressor_logits = nn.Parameter(logits)
            self.register_buffer("compressor_lut", torch.empty(0), persistent=False)
        else:
            self.register_buffer("compressor_lut", lut.float(), persistent=True)
            self.compressor_logits = None

        self.metric_head = nn.Sequential(
            nn.Linear(int(num_slots) * int(num_tokens), 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, len(DMULT_TARGET_COLUMNS)),
        )

    @staticmethod
    def _and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def _or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - (a * b)

    @staticmethod
    def _xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - (2.0 * a * b)

    def _ha(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._xor(a, b), self._and(a, b)

    def _fa(self, a: torch.Tensor, b: torch.Tensor, cin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._xor(self._xor(a, b), cin)
        carry = (a * b) + (a * cin) + (b * cin) - (2.0 * a * b * cin)
        return s, carry

    def _e42(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        cin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        axb = self._xor(a, b)
        part = self._xor(self._xor(axb, c), d)
        s = self._xor(part, cin)
        cout = self._or(self._and(axb, c), self._and(1.0 - axb, a))
        carry = self._or(self._and(part, cin), self._and(1.0 - part, d))
        return s, carry, cout

    def _ae42_or_exact(
        self,
        token: torch.Tensor | None,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        cin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exact_s, exact_c, exact_cout = self._e42(a, b, c, d, cin)
        if token is None:
            return exact_s, exact_c, exact_cout
        s0, c0 = self._lut4(token, a, b, c, d)
        wrapped_s = self._xor(s0, cin)
        wrapped_cout = self._and(s0, cin)
        wrapped_c = c0
        mask = (token.long() > 0).to(dtype=a.dtype, device=a.device).reshape(-1, 1)
        return (
            (exact_s * (1.0 - mask)) + (wrapped_s * mask),
            (exact_c * (1.0 - mask)) + (wrapped_c * mask),
            (exact_cout * (1.0 - mask)) + (wrapped_cout * mask),
        )

    def _lut_probs(self) -> torch.Tensor:
        if self.compressor_logits is not None:
            return torch.sigmoid(self.compressor_logits)
        return self.compressor_lut

    def _lut4(
        self,
        token: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = self._lut_probs()[token.long().clamp(min=0, max=self.num_tokens - 1)]
        weights = torch.stack(
            [
                (1.0 - a) * (1.0 - b) * (1.0 - c) * (1.0 - d),
                (1.0 - a) * (1.0 - b) * (1.0 - c) * d,
                (1.0 - a) * (1.0 - b) * c * (1.0 - d),
                (1.0 - a) * (1.0 - b) * c * d,
                (1.0 - a) * b * (1.0 - c) * (1.0 - d),
                (1.0 - a) * b * (1.0 - c) * d,
                (1.0 - a) * b * c * (1.0 - d),
                (1.0 - a) * b * c * d,
                a * (1.0 - b) * (1.0 - c) * (1.0 - d),
                a * (1.0 - b) * (1.0 - c) * d,
                a * (1.0 - b) * c * (1.0 - d),
                a * (1.0 - b) * c * d,
                a * b * (1.0 - c) * (1.0 - d),
                a * b * (1.0 - c) * d,
                a * b * c * (1.0 - d),
                a * b * c * d,
            ],
            dim=-1,
        )
        out = torch.einsum("bnk,bko->bno", weights, table)
        return out[..., 0], out[..., 1]

    def _pp(self, i: int, j: int, batch: int) -> torch.Tensor:
        return (self.a_bits[int(i)] * self.b_bits[int(j)]).unsqueeze(0).expand(batch, -1)

    def _zero(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.a_bits.shape[1], dtype=self.a_bits.dtype, device=self.a_bits.device)

    def forward(self, slot_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = slot_tokens.long().clamp(min=0, max=self.num_tokens - 1)
        if int(tokens.shape[1]) != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} slot tokens, got {int(tokens.shape[1])}.")
        batch = int(tokens.shape[0])
        z = self._zero(batch)
        high_tokens = tokens[:, 9:17] if self.num_slots == 17 else None

        pp1: dict[int, list[torch.Tensor]] = {
            0: [self._pp(0, 0, batch)],
            1: [self._pp(1, 0, batch), self._pp(0, 1, batch)],
            2: [self._pp(2, 0, batch), self._pp(1, 1, batch), self._pp(0, 2, batch)],
            3: [self._pp(3, 0, batch), self._pp(2, 1, batch), self._pp(1, 2, batch), self._pp(0, 3, batch)],
        }

        s4ha, c4ha = self._ha(self._pp(0, 4, batch), self._pp(1, 3, batch))
        pp1[4] = [s4ha, self._pp(4, 0, batch), self._pp(3, 1, batch), self._pp(2, 2, batch)]

        s5, c5 = self._lut4(tokens[:, 0], self._pp(0, 5, batch), self._pp(1, 4, batch), self._pp(2, 3, batch), self._pp(3, 2, batch))
        pp1[5] = [c4ha, s5, self._pp(5, 0, batch), self._pp(4, 1, batch)]

        s6, c6 = self._lut4(tokens[:, 1], self._pp(0, 6, batch), self._pp(1, 5, batch), self._pp(2, 4, batch), self._pp(3, 3, batch))
        s6ha, c6ha = self._ha(self._pp(4, 2, batch), self._pp(5, 1, batch))
        pp1[6] = [c5, s6ha, s6, self._pp(6, 0, batch)]

        s70, c70 = self._lut4(tokens[:, 2], self._pp(0, 7, batch), self._pp(1, 6, batch), self._pp(2, 5, batch), self._pp(3, 4, batch))
        s71, c71 = self._lut4(tokens[:, 3], self._pp(4, 3, batch), self._pp(5, 2, batch), self._pp(6, 1, batch), self._pp(7, 0, batch))
        pp1[7] = [c6ha, c6, s71, s70]

        s8, c8, co8 = self._ae42_or_exact(
            None if high_tokens is None else high_tokens[:, 0],
            self._pp(1, 7, batch),
            self._pp(2, 6, batch),
            self._pp(3, 5, batch),
            self._pp(4, 4, batch),
            z,
        )
        s8fa, c8fa = self._fa(self._pp(5, 3, batch), self._pp(6, 2, batch), self._pp(7, 1, batch))
        pp1[8] = [c71, c70, s8fa, s8]

        s9, c9, co9 = self._ae42_or_exact(
            None if high_tokens is None else high_tokens[:, 1],
            self._pp(2, 7, batch),
            self._pp(3, 6, batch),
            self._pp(4, 5, batch),
            self._pp(5, 4, batch),
            co8,
        )
        s9ha, c9ha = self._ha(self._pp(6, 3, batch), self._pp(7, 2, batch))
        pp1[9] = [c8fa, c8, s9ha, s9]

        s10, c10, co10 = self._ae42_or_exact(
            None if high_tokens is None else high_tokens[:, 2],
            self._pp(3, 7, batch),
            self._pp(4, 6, batch),
            self._pp(5, 5, batch),
            self._pp(6, 4, batch),
            co9,
        )
        pp1[10] = [c9ha, c9, s10, self._pp(7, 3, batch)]

        s11fa, c11fa = self._fa(self._pp(4, 7, batch), self._pp(5, 6, batch), co10)
        pp1[11] = [c10, self._pp(7, 4, batch), self._pp(6, 5, batch), s11fa]
        pp1[12] = [c11fa, self._pp(7, 5, batch), self._pp(6, 6, batch), self._pp(5, 7, batch)]
        pp1[13] = [self._pp(7, 6, batch), self._pp(6, 7, batch)]
        pp1[14] = [self._pp(7, 7, batch)]

        ha2s, ha2c = self._ha(pp1[2][0], pp1[2][1])
        c3s, c3c = self._lut4(tokens[:, 4], pp1[3][0], pp1[3][1], pp1[3][2], pp1[3][3])
        c4s, c4c = self._lut4(tokens[:, 5], pp1[4][0], pp1[4][1], pp1[4][2], pp1[4][3])
        c5s, c5c = self._lut4(tokens[:, 6], pp1[5][0], pp1[5][1], pp1[5][2], pp1[5][3])
        c6s, c6c = self._lut4(tokens[:, 7], pp1[6][0], pp1[6][1], pp1[6][2], pp1[6][3])
        c7s, c7c = self._lut4(tokens[:, 8], pp1[7][0], pp1[7][1], pp1[7][2], pp1[7][3])

        e8s, e8c, e8co = self._ae42_or_exact(None if high_tokens is None else high_tokens[:, 3], pp1[8][0], pp1[8][1], pp1[8][2], pp1[8][3], z)
        e9s, e9c, e9co = self._ae42_or_exact(None if high_tokens is None else high_tokens[:, 4], pp1[9][0], pp1[9][1], pp1[9][2], pp1[9][3], e8co)
        e10s, e10c, e10co = self._ae42_or_exact(None if high_tokens is None else high_tokens[:, 5], pp1[10][0], pp1[10][1], pp1[10][2], pp1[10][3], e9co)
        e11s, e11c, e11co = self._ae42_or_exact(None if high_tokens is None else high_tokens[:, 6], pp1[11][0], pp1[11][1], pp1[11][2], pp1[11][3], e10co)
        e12s, e12c, e12co = self._ae42_or_exact(None if high_tokens is None else high_tokens[:, 7], pp1[12][0], pp1[12][1], pp1[12][2], pp1[12][3], e11co)
        fa13s, fa13c = self._fa(pp1[13][0], pp1[13][1], e12co)

        a_row = [pp1[0][0], pp1[1][0], ha2s, c3s, c4s, c5s, c6s, c7s, e8s, e9s, e10s, e11s, e12s, fa13s, pp1[14][0]]
        b_row = [z, pp1[1][1], pp1[2][2], ha2c, c3c, c4c, c5c, c6c, c7c, e8c, e9c, e10c, e11c, e12c, fa13c]
        truth_flat = torch.zeros(batch, self.a_bits.shape[1], dtype=self.a_bits.dtype, device=self.a_bits.device)
        for bit_idx, (a_bit, b_bit) in enumerate(zip(a_row, b_row)):
            truth_flat = truth_flat + (float(1 << bit_idx) * (a_bit + b_bit))
        truth_flat = truth_flat.masked_fill(self.zero_operand_mask_flat.unsqueeze(0), 0.0)
        delta_flat = truth_flat - self.exact_flat.unsqueeze(0)

        onehot = F.one_hot(tokens, num_classes=self.num_tokens).float().reshape(batch, -1)
        metric_pred = self.metric_head(onehot)
        return {
            "delta_flat": delta_flat,
            "delta": delta_flat.reshape(batch, TABLE_SIDE, TABLE_SIDE),
            "metric_pred": metric_pred,
        }


def metric_from_delta_torch(
    delta: torch.Tensor,
    *,
    exact: torch.Tensor,
    nonzero_mask: torch.Tensor,
    soft_er: bool = True,
) -> torch.Tensor:
    err = delta.float()
    ed = torch.abs(err)
    re = torch.zeros_like(err)
    red = torch.zeros_like(err)
    re[:, nonzero_mask] = err[:, nonzero_mask] / exact[nonzero_mask].unsqueeze(0)
    red[:, nonzero_mask] = ed[:, nonzero_mask] / exact[nonzero_mask].unsqueeze(0)
    mu_e = err.mean(dim=(1, 2))
    mu_ed = ed.mean(dim=(1, 2))
    mu_re = re.mean(dim=(1, 2))
    mu_red = red.mean(dim=(1, 2))
    sigma_e = (err * err).mean(dim=(1, 2)) - (mu_e * mu_e)
    sigma_ed = (ed * ed).mean(dim=(1, 2)) - (mu_ed * mu_ed)
    sigma_re = (re * re).mean(dim=(1, 2)) - (mu_re * mu_re)
    sigma_red = (red * red).mean(dim=(1, 2)) - (mu_red * mu_red)
    rms_e = torch.sqrt((err * err).mean(dim=(1, 2)) + 1.0e-8)
    rms_re = torch.sqrt((re * re).mean(dim=(1, 2)) + 1.0e-8)
    if bool(soft_er):
        er_value = torch.sigmoid((torch.abs(err) - 0.5) / 0.25).mean(dim=(1, 2))
    else:
        er_value = (torch.abs(err) >= 0.5).float().mean(dim=(1, 2))
    return torch.stack(
        [
            mu_e,
            sigma_e,
            mu_ed,
            mu_ed / float(M_MAX_U8_MUL),
            sigma_ed,
            mu_re,
            sigma_re,
            mu_red,
            sigma_red,
            rms_e,
            rms_re,
            er_value,
        ],
        dim=1,
    )


