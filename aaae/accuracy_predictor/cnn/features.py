from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .contracts import (
    ApplicationProfile,
    DataContractError,
    DomainMismatchError,
    FeatureTable,
    MultiplierDesignBatch,
)

_DOMAINS = frozenset({"u8_numeric", "s8_numeric", "s8_twos_complement"})
_EVIDENCE_KINDS = frozenset({"measured", "synthetic"})
_DESIGN_CHUNK_SIZE = 64
DEFAULT_FEATURE_SCHEMA = "cnn_ame_basis_v1"
RESNET_STAGE_FEATURE_SCHEMA = "cnn_ame_basis32_stage_v3"
MOBILENET_STAGE_FEATURE_SCHEMA = "cnn_ame_basis32_mobilenet_v1"
FEATURE_SCHEMA_BINS = {
    DEFAULT_FEATURE_SCHEMA: 4,
    "cnn_ame_basis32_v2": 32,
    RESNET_STAGE_FEATURE_SCHEMA: 32,
    MOBILENET_STAGE_FEATURE_SCHEMA: 32,
}
SUPPORTED_FEATURE_SCHEMAS = frozenset(FEATURE_SCHEMA_BINS)
STAGE_ROLE_FEATURE_SCHEMAS = frozenset(
    {RESNET_STAGE_FEATURE_SCHEMA, MOBILENET_STAGE_FEATURE_SCHEMA}
)
_STAGE_ROLE_STATISTICS = (
    "signed_ame_log1p",
    "absolute_ame_log1p",
    "positive_ame_log1p",
    "negative_ame_log1p",
    "rms_error_log1p",
    "nonzero_rate_log1p",
)


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _semantic_order(domain: str) -> np.ndarray:
    if domain == "s8_twos_complement":
        return np.concatenate(
            [np.arange(128, 256, dtype=np.int64), np.arange(0, 128, dtype=np.int64)]
        )
    return np.arange(256, dtype=np.int64)


def _numeric_array(values: object, name: str, dtype: np.dtype) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise DataContractError(f"{name} must be a numeric array") from error
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise DataContractError(f"{name} must be a real numeric array")
    try:
        return np.asarray(array, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as error:
        raise DataContractError(
            f"{name} cannot be represented as {np.dtype(dtype).name}"
        ) from error


def _prepare_profile(
    profile: ApplicationProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if profile.domain not in _DOMAINS:
        raise DataContractError(f"unsupported arithmetic domain: {profile.domain}")
    if not isinstance(profile.network_id, str) or not profile.network_id:
        raise DataContractError("network_id must be a nonempty string")
    if not isinstance(profile.dataset_id, str) or not profile.dataset_id:
        raise DataContractError("dataset_id must be a nonempty string")
    if profile.profile_evidence not in _EVIDENCE_KINDS:
        raise DataContractError(
            "profile_evidence must be one of: measured, synthetic; "
            f"got {profile.profile_evidence}"
        )

    layer_count = len(profile.layer_ids)
    if layer_count == 0:
        raise DataContractError("profile must contain at least one layer")
    if any(
        not isinstance(layer_id, str) or not layer_id for layer_id in profile.layer_ids
    ):
        raise DataContractError("layer_ids must be nonempty strings")
    if len(set(profile.layer_ids)) != layer_count:
        raise DataContractError("layer_ids must be unique")

    activation = _numeric_array(
        profile.activation_histograms,
        "activation_histograms",
        np.dtype(np.float64),
    )
    weights = _numeric_array(
        profile.weight_histograms,
        "weight_histograms",
        np.dtype(np.float64),
    )
    expected_histogram_shape = (layer_count, 256)
    for name, histograms in (
        ("activation_histograms", activation),
        ("weight_histograms", weights),
    ):
        if histograms.shape != expected_histogram_shape:
            raise DataContractError(
                f"{name} must have shape [L, 256]; got {histograms.shape}"
            )
        if not np.isfinite(histograms).all():
            raise DataContractError(f"{name} must contain only finite values")
        if np.any(histograms < 0):
            raise DataContractError(f"{name} must be nonnegative")
        if np.any(histograms.sum(axis=1) <= 0):
            raise DataContractError(f"{name} rows must have positive mass")

    mac_values = _numeric_array(profile.mac_counts, "mac_counts", np.dtype(np.float64))
    if mac_values.shape != (layer_count,):
        raise DataContractError(
            f"mac_counts must have shape [L]; got {mac_values.shape}"
        )
    if not np.isfinite(mac_values).all():
        raise DataContractError("mac_counts must contain only finite values")
    if np.any(mac_values != np.floor(mac_values)):
        raise DataContractError("mac_counts must contain integers")
    if np.any(mac_values <= 0):
        raise DataContractError("mac_counts must be positive")
    try:
        mac_counts = mac_values.astype(np.int64)
    except (OverflowError, ValueError) as error:
        raise DataContractError("mac_counts cannot be represented as int64") from error
    if np.any(mac_counts.astype(np.float64) != mac_values):
        raise DataContractError("mac_counts cannot be represented as int64")

    propagation_gains = _numeric_array(
        profile.propagation_gains,
        "propagation_gains",
        np.dtype(np.float64),
    )
    if propagation_gains.shape != (layer_count,):
        raise DataContractError(
            f"propagation_gains must have shape [L]; got {propagation_gains.shape}"
        )
    if not np.isfinite(propagation_gains).all():
        raise DataContractError("propagation_gains must contain only finite values")
    if np.any(propagation_gains < 0):
        raise DataContractError("propagation_gains must be nonnegative")
    if not np.any(propagation_gains > 0):
        raise DataContractError(
            "propagation_gains must contain at least one positive value"
        )

    activation = activation / activation.sum(axis=1, keepdims=True)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return activation, weights, mac_counts, propagation_gains


def _profile_digest(
    profile: ApplicationProfile,
    normalized_activation: np.ndarray,
    normalized_weights: np.ndarray,
    mac_counts: np.ndarray,
    propagation_gains: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "dataset_id": profile.dataset_id,
        "domain": profile.domain,
        "layer_ids": list(profile.layer_ids),
        "network_id": profile.network_id,
        "profile_evidence": profile.profile_evidence,
        "schema": "cnn_application_profile_digest_v1",
    }
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    canonical_arrays = (
        ("activation_histograms", normalized_activation, "<f8"),
        ("weight_histograms", normalized_weights, "<f8"),
        ("mac_counts", mac_counts, "<i8"),
        ("propagation_gains", propagation_gains, "<f8"),
    )
    for name, values, dtype in canonical_arrays:
        array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
        digest.update(name.encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _feature_names(bin_count: int) -> tuple[str, ...]:
    names = [
        "paper_ame_signed_log1p",
        "weighted_absolute_error_log1p",
        "weighted_rms_error_log1p",
        "weighted_positive_error_log1p",
        "weighted_negative_error_log1p",
        "weighted_nonzero_rate_log1p",
        "global_mean_error_signed_log1p",
        "global_mean_absolute_error_log1p",
        "global_rmse_log1p",
        "global_worst_absolute_error_log1p",
        "global_error_rate_log1p",
    ]
    for activation_bin in range(bin_count):
        for weight_bin in range(bin_count):
            names.append(f"basis_a{activation_bin}_w{weight_bin}_signed_ame_log1p")
    for activation_bin in range(bin_count):
        for weight_bin in range(bin_count):
            names.append(
                f"basis_a{activation_bin}_w{weight_bin}_weighted_magnitude_log1p"
            )
    for activation_bin in range(bin_count):
        for weight_bin in range(bin_count):
            names.append(
                f"basis_a{activation_bin}_w{weight_bin}_weighted_nonzero_rate_log1p"
            )
    return tuple(names)


def _resnet_stage_role(layer_id: str) -> str:
    if layer_id == "/conv1/Conv":
        return "stem"
    stage_match = re.search(r"/(layer[0-9]+)/", layer_id)
    if stage_match is None:
        raise DataContractError(
            "cnn_ame_basis32_stage_v3 requires ResNet layer IDs; "
            f"cannot identify a residual stage for {layer_id!r}"
        )
    if "/downsample/" in layer_id:
        role = "downsample"
    else:
        role_match = re.search(r"/(conv[0-9]+)/", layer_id)
        if role_match is None:
            raise DataContractError(
                "cnn_ame_basis32_stage_v3 requires ResNet layer IDs; "
                f"cannot identify a convolution role for {layer_id!r}"
            )
        role = role_match.group(1)
    return f"{stage_match.group(1)}_{role}"


def _mobilenet_stage_role(layer_id: str) -> str:
    if layer_id == "/features/features.0/features.0.0/Conv":
        return "stem"
    if layer_id == "/features/features.16/features.16.0/Conv":
        return "final"

    match = re.match(
        r"^/features/features\.([0-9]+)/block/block\.([0-9]+)/",
        layer_id,
    )
    if match is None:
        raise DataContractError(
            f"{MOBILENET_STAGE_FEATURE_SCHEMA} requires MobileNetV3 layer IDs; "
            f"cannot identify an inverted-residual block for {layer_id!r}"
        )
    feature_index = int(match.group(1))
    block_index = int(match.group(2))
    if feature_index == 1:
        stage = "stage1"
        roles = {0: "depthwise", 1: "project"}
    elif feature_index in {2, 3}:
        stage = "stage2"
        roles = {0: "expand", 1: "depthwise", 2: "project"}
    elif feature_index in {4, 5, 6}:
        stage = "stage3"
        roles = {0: "expand", 1: "depthwise", 3: "project"}
    elif feature_index in {7, 8, 9, 10}:
        stage = "stage4"
        roles = {0: "expand", 1: "depthwise", 2: "project"}
    elif feature_index in {11, 12}:
        stage = "stage5"
        roles = {0: "expand", 1: "depthwise", 3: "project"}
    elif feature_index in {13, 14, 15}:
        stage = "stage6"
        roles = {0: "expand", 1: "depthwise", 3: "project"}
    else:
        raise DataContractError(
            f"{MOBILENET_STAGE_FEATURE_SCHEMA} requires MobileNetV3-Large "
            f"feature indices 1..15; got {feature_index} in {layer_id!r}"
        )

    role = roles.get(block_index)
    if (
        role is None
        and block_index == 2
        and feature_index
        in {
            4,
            5,
            6,
            11,
            12,
            13,
            14,
            15,
        }
    ):
        if "/fc1/" in layer_id:
            role = "se_reduce"
        elif "/fc2/" in layer_id:
            role = "se_expand"
    if role is None:
        raise DataContractError(
            f"{MOBILENET_STAGE_FEATURE_SCHEMA} cannot identify a Conv role "
            f"for {layer_id!r}"
        )
    return f"{stage}_{role}"


def _stage_role_application_matrices(
    layer_ids: tuple[str, ...],
    activation: np.ndarray,
    weights: np.ndarray,
    mac_counts: np.ndarray,
    propagation_gains: np.ndarray,
    *,
    feature_schema: str,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if feature_schema == RESNET_STAGE_FEATURE_SCHEMA:
        role_resolver = _resnet_stage_role
    elif feature_schema == MOBILENET_STAGE_FEATURE_SCHEMA:
        role_resolver = _mobilenet_stage_role
    else:
        raise DataContractError(
            f"feature schema {feature_schema!r} has no stage-role resolver"
        )
    group_names: list[str] = []
    group_indices: list[int] = []
    for layer_id in layer_ids:
        group = role_resolver(layer_id)
        if group not in group_names:
            group_names.append(group)
        group_indices.append(group_names.index(group))

    matrices = np.zeros((len(group_names), 256, 256), dtype=np.float64)
    coefficients = mac_counts.astype(np.float64) * propagation_gains
    for layer_index, group_index in enumerate(group_indices):
        matrices[group_index] += coefficients[layer_index] * np.outer(
            activation[layer_index],
            weights[layer_index],
        )
    masses = matrices.sum(axis=(1, 2))
    if not np.isfinite(matrices).all() or np.any(masses <= 0.0):
        raise DataContractError(
            f"{feature_schema} stage-role application matrices must have positive finite mass"
        )
    return tuple(group_names), matrices, masses


def _stage_role_feature_names(group_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"stage_role_{group}_{statistic}"
        for group in group_names
        for statistic in _STAGE_ROLE_STATISTICS
    )


def _build_stage_role_chunk(
    delta: np.ndarray,
    matrix_columns: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    flattened = delta.reshape(len(delta), -1)
    absolute_delta = np.abs(flattened)
    signed = flattened @ matrix_columns
    absolute = absolute_delta @ matrix_columns
    squared = (flattened * flattened) @ matrix_columns
    nonzero = (flattened != 0) @ matrix_columns
    positive = np.maximum((absolute + signed) * 0.5, 0.0)
    negative = np.maximum((absolute - signed) * 0.5, 0.0)
    values = np.stack(
        (
            _signed_log1p(signed),
            np.log1p(absolute),
            np.log1p(positive),
            np.log1p(negative),
            np.log1p(np.sqrt(squared / masses[None, :])),
            np.log1p(nonzero / masses[None, :]),
        ),
        axis=2,
    )
    return values.reshape(len(delta), -1).astype(np.float32)


def _build_chunk(
    delta: np.ndarray,
    application_matrix: np.ndarray,
    magnitude_matrix: np.ndarray,
    magnitude_mass: float,
    semantic_order: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    absolute_delta = np.abs(delta)
    paper_ame = np.einsum("nij,ij->n", delta, application_matrix, optimize=True)
    weighted_absolute = np.einsum(
        "nij,ij->n",
        absolute_delta,
        magnitude_matrix,
        optimize=True,
    )
    weighted_rms = np.sqrt(
        np.einsum("nij,ij->n", delta * delta, magnitude_matrix, optimize=True)
        / magnitude_mass
    )
    weighted_positive = np.einsum(
        "nij,ij->n",
        np.maximum(delta, 0.0),
        magnitude_matrix,
        optimize=True,
    )
    weighted_negative = np.einsum(
        "nij,ij->n",
        np.maximum(-delta, 0.0),
        magnitude_matrix,
        optimize=True,
    )
    weighted_nonzero = (
        np.einsum(
            "nij,ij->n",
            delta != 0,
            magnitude_matrix,
            optimize=True,
        )
        / magnitude_mass
    )

    columns = [
        _signed_log1p(paper_ame),
        np.log1p(weighted_absolute),
        np.log1p(weighted_rms),
        np.log1p(weighted_positive),
        np.log1p(weighted_negative),
        np.log1p(weighted_nonzero),
        _signed_log1p(delta.mean(axis=(1, 2))),
        np.log1p(absolute_delta.mean(axis=(1, 2))),
        np.log1p(np.sqrt(np.mean(delta * delta, axis=(1, 2)))),
        np.log1p(absolute_delta.max(axis=(1, 2))),
        np.log1p(np.mean(delta != 0, axis=(1, 2))),
    ]

    bin_width = 256 // bin_count
    ordered_delta = delta[:, semantic_order][:, :, semantic_order]
    ordered_application = application_matrix[semantic_order][:, semantic_order]
    ordered_magnitude = magnitude_matrix[semantic_order][:, semantic_order]
    blocked_delta = ordered_delta.reshape(
        len(delta), bin_count, bin_width, bin_count, bin_width
    )
    blocked_application = ordered_application.reshape(
        bin_count, bin_width, bin_count, bin_width
    )
    blocked_magnitude = ordered_magnitude.reshape(
        bin_count, bin_width, bin_count, bin_width
    )
    signed_basis = np.einsum(
        "naibj,aibj->nab",
        blocked_delta,
        blocked_application,
        optimize=True,
    ).reshape(len(delta), -1)
    magnitude_basis = np.einsum(
        "naibj,aibj->nab",
        np.abs(blocked_delta),
        blocked_magnitude,
        optimize=True,
    ).reshape(len(delta), -1)
    nonzero_basis = (
        np.einsum(
            "naibj,aibj->nab",
            blocked_delta != 0,
            blocked_magnitude,
            optimize=True,
        ).reshape(len(delta), -1)
        / magnitude_mass
    )
    columns.extend(
        _signed_log1p(signed_basis[:, index]) for index in range(bin_count**2)
    )
    columns.extend(np.log1p(magnitude_basis[:, index]) for index in range(bin_count**2))
    columns.extend(np.log1p(nonzero_basis[:, index]) for index in range(bin_count**2))
    return np.column_stack(columns).astype(np.float32)


def build_features(
    profile: ApplicationProfile,
    designs: MultiplierDesignBatch,
    *,
    chunk_designs: int = _DESIGN_CHUNK_SIZE,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> FeatureTable:
    if (
        isinstance(chunk_designs, bool)
        or not isinstance(chunk_designs, int)
        or chunk_designs < 1
    ):
        raise DataContractError("chunk_designs must be a positive integer")
    if feature_schema not in SUPPORTED_FEATURE_SCHEMAS:
        raise DataContractError(
            f"unsupported CNN feature schema: {feature_schema!r}; "
            f"expected one of {sorted(SUPPORTED_FEATURE_SCHEMAS)}"
        )
    bin_count = FEATURE_SCHEMA_BINS[feature_schema]
    activation, weights, mac_counts, propagation_gains = _prepare_profile(profile)
    if designs.domain not in _DOMAINS:
        raise DataContractError(f"unsupported arithmetic domain: {designs.domain}")
    if profile.domain != designs.domain:
        raise DomainMismatchError(
            f"profile domain {profile.domain} does not match design domain {designs.domain}"
        )
    if len(set(designs.design_ids)) != len(designs.design_ids):
        raise DataContractError("design_ids must be unique")
    if not designs.design_ids:
        raise DataContractError("design batch must contain at least one design")
    if any(
        not isinstance(design_id, str) or not design_id
        for design_id in designs.design_ids
    ):
        raise DataContractError("design_ids must be nonempty strings")
    try:
        delta_input = np.asarray(designs.delta)
    except (TypeError, ValueError) as error:
        raise DataContractError("delta must be a numeric array") from error
    if not np.issubdtype(delta_input.dtype, np.number) or np.issubdtype(
        delta_input.dtype,
        np.complexfloating,
    ):
        raise DataContractError("delta must be a real numeric array")
    if delta_input.ndim != 3 or delta_input.shape[1:] != (256, 256):
        raise DataContractError(
            f"delta must have shape [N, 256, 256]; got {delta_input.shape}"
        )
    if delta_input.shape[0] != len(designs.design_ids):
        raise DataContractError(
            "delta row count must match design_ids; "
            f"got {delta_input.shape[0]} rows and {len(designs.design_ids)} IDs"
        )

    coefficients = mac_counts.astype(np.float64) * propagation_gains
    application_matrix = np.einsum(
        "l,li,lj->ij",
        coefficients,
        activation,
        weights,
        optimize=True,
    )
    if not np.isfinite(application_matrix).all():
        raise DataContractError("application matrix must contain only finite values")
    magnitude_matrix = np.abs(application_matrix)
    magnitude_mass = float(magnitude_matrix.sum())
    if not np.isfinite(magnitude_mass) or magnitude_mass <= 0:
        raise DataContractError(
            "application matrix must have positive finite magnitude"
        )

    stage_role_matrix_columns: np.ndarray | None = None
    stage_role_masses: np.ndarray | None = None
    feature_names = _feature_names(bin_count)
    if feature_schema in STAGE_ROLE_FEATURE_SCHEMAS:
        group_names, stage_role_matrices, stage_role_masses = (
            _stage_role_application_matrices(
                profile.layer_ids,
                activation,
                weights,
                mac_counts,
                propagation_gains,
                feature_schema=feature_schema,
            )
        )
        stage_role_matrix_columns = np.ascontiguousarray(
            stage_role_matrices.reshape(len(stage_role_matrices), -1).T
        )
        feature_names += _stage_role_feature_names(group_names)
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError("CNN feature schema contains duplicate feature names")
    values = np.empty((len(designs.design_ids), len(feature_names)), dtype=np.float32)
    semantic_order = _semantic_order(profile.domain)
    for start in range(0, len(designs.design_ids), chunk_designs):
        stop = min(start + chunk_designs, len(designs.design_ids))
        try:
            delta_chunk = np.asarray(delta_input[start:stop], dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as error:
            raise DataContractError(
                f"delta chunk [{start}:{stop}] cannot be represented as float64"
            ) from error
        finite_rows = np.isfinite(delta_chunk).all(axis=(1, 2))
        if not np.all(finite_rows):
            bad_offset = int(np.flatnonzero(~finite_rows)[0])
            bad_id = designs.design_ids[start + bad_offset]
            raise DataContractError(
                f"delta contains non-finite values for design {bad_id}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            chunk_features = _build_chunk(
                delta_chunk,
                application_matrix,
                magnitude_matrix,
                magnitude_mass,
                semantic_order,
                bin_count,
            )
            if stage_role_matrix_columns is not None and stage_role_masses is not None:
                chunk_features = np.column_stack(
                    (
                        chunk_features,
                        _build_stage_role_chunk(
                            delta_chunk,
                            stage_role_matrix_columns,
                            stage_role_masses,
                        ),
                    )
                ).astype(np.float32, copy=False)
        finite_feature_rows = np.isfinite(chunk_features).all(axis=1)
        if not np.all(finite_feature_rows):
            bad_offset = int(np.flatnonzero(~finite_feature_rows)[0])
            bad_id = designs.design_ids[start + bad_offset]
            raise DataContractError(
                f"computed features are non-finite for design {bad_id}"
            )
        values[start:stop] = chunk_features

    return FeatureTable(
        design_ids=tuple(designs.design_ids),
        feature_names=feature_names,
        values=values,
        schema_name=feature_schema,
        profile_digest=_profile_digest(
            profile,
            activation,
            weights,
            mac_counts,
            propagation_gains,
        ),
    )
