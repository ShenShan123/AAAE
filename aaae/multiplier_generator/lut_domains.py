from __future__ import annotations

import numpy as np


TABLE_SIDE = 256
S8_QMIN = -128
S8_QMAX = 127
S8_MAX_ABS_PRODUCT = abs(S8_QMIN) * abs(S8_QMIN)


def s8_code_axis_np() -> np.ndarray:
    """Return the signed code represented by each S8S8 LUT index."""
    return np.arange(S8_QMIN, S8_QMAX + 1, dtype=np.int32)


def s8s8_exact_table_np() -> np.ndarray:
    """Return the exact signed product table indexed by code + 128."""
    codes = s8_code_axis_np()
    return codes[:, None] * codes[None, :]


def u8_lut_to_s8s8_sign_magnitude(u8_lut: np.ndarray) -> np.ndarray:
    """Expand an unsigned magnitude LUT into the requested S8S8 sign-magnitude LUT.

    The final two dimensions of ``u8_lut`` are indexed by unsigned magnitudes.
    The returned table is indexed by ``signed_code + 128`` and defines:

        signed_lut[q_a + 128, q_b + 128]
        = sign(q_a) * sign(q_b) * u8_lut[abs(q_a), abs(q_b)]

    This is a signed wrapper around the existing unsigned approximate core, not
    a claim that the source RTL implements native two's-complement arithmetic.
    """
    source = np.asarray(u8_lut)
    if source.ndim < 2 or tuple(source.shape[-2:]) != (TABLE_SIDE, TABLE_SIDE):
        raise ValueError(
            f"u8_lut must end in {(TABLE_SIDE, TABLE_SIDE)}, got shape={source.shape}"
        )
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"u8_lut must use an integer dtype, got {source.dtype}")

    codes = s8_code_axis_np()
    magnitudes = np.abs(codes)
    signs = np.sign(codes).astype(np.int32)
    magnitude_lut = np.asarray(
        source[..., magnitudes[:, None], magnitudes[None, :]],
        dtype=np.int32,
    )
    return magnitude_lut * (signs[:, None] * signs[None, :])


def s8s8_metric12_from_lut(s8s8_lut: np.ndarray) -> np.ndarray:
    """Compute the existing 12 multiplier metrics against signed S8S8 exact values.

    The final two dimensions must be LUT axes indexed by ``signed_code + 128``.
    Leading dimensions, when present, are treated as independent tables.  The
    output's final dimension follows ``DMULT_TARGET_COLUMNS`` ordering.
    """
    table = np.asarray(s8s8_lut)
    if table.ndim < 2 or tuple(table.shape[-2:]) != (TABLE_SIDE, TABLE_SIDE):
        raise ValueError(
            f"s8s8_lut must end in {(TABLE_SIDE, TABLE_SIDE)}, got shape={table.shape}"
        )
    if not np.issubdtype(table.dtype, np.integer):
        raise TypeError(f"s8s8_lut must use an integer dtype, got {table.dtype}")

    exact = s8s8_exact_table_np().astype(np.float64, copy=False)
    error = table.astype(np.float64) - exact
    absolute_error = np.abs(error)
    nonzero = exact != 0.0
    relative_error = np.zeros_like(error)
    relative_absolute_error = np.zeros_like(error)
    np.divide(error, exact, out=relative_error, where=nonzero)
    np.divide(absolute_error, np.abs(exact), out=relative_absolute_error, where=nonzero)

    reduce_axes = (-2, -1)
    mu_e = error.mean(axis=reduce_axes)
    mu_ed = absolute_error.mean(axis=reduce_axes)
    mu_re = relative_error.mean(axis=reduce_axes)
    mu_red = relative_absolute_error.mean(axis=reduce_axes)
    sigma_e = (error * error).mean(axis=reduce_axes) - (mu_e * mu_e)
    sigma_ed = (absolute_error * absolute_error).mean(axis=reduce_axes) - (mu_ed * mu_ed)
    sigma_re = (relative_error * relative_error).mean(axis=reduce_axes) - (mu_re * mu_re)
    sigma_red = (relative_absolute_error * relative_absolute_error).mean(axis=reduce_axes) - (mu_red * mu_red)
    rms_e = np.sqrt((error * error).mean(axis=reduce_axes))
    rms_re = np.sqrt((relative_error * relative_error).mean(axis=reduce_axes))
    er = (absolute_error >= 0.5).mean(axis=reduce_axes)
    return np.stack(
        (
            mu_e,
            sigma_e,
            mu_ed,
            mu_ed / float(S8_MAX_ABS_PRODUCT),
            sigma_ed,
            mu_re,
            sigma_re,
            mu_red,
            sigma_red,
            rms_e,
            rms_re,
            er,
        ),
        axis=-1,
    ).astype(np.float32)
