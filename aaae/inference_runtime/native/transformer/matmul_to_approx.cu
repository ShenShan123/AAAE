// ONNX Runtime custom op (CUDA): ApproxMatMulCuda
#include <cuda_runtime.h>
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cnn_cuda_ops/cuda_approx/Akbar_1.cuh"
#include "../cnn_cuda_ops/cuda_approx/Akbar_2.cuh"
#include "../cnn_cuda_ops/cuda_approx/Antonio.cuh"
#include "../cnn_cuda_ops/cuda_approx/Ha.cuh"
#include "../cnn_cuda_ops/cuda_approx/Hybrid_1.cuh"
#include "../cnn_cuda_ops/cuda_approx/Hybrid_2.cuh"
#include "../cnn_cuda_ops/cuda_approx/MC16.cuh"
#include "../cnn_cuda_ops/cuda_approx/momeni.cuh"
#include "../cnn_cuda_ops/cuda_approx/SabetZ.cuh"
#include "../cnn_cuda_ops/cuda_approx/AP1.cuh"
#include "../cnn_cuda_ops/cuda_approx/AP2.cuh"
#include "../cnn_cuda_ops/cuda_approx/yang1.cuh"
#include "../cnn_cuda_ops/cuda_approx/yang2.cuh"
#include "../cnn_cuda_ops/cuda_approx/yang3.cuh"
#ifdef EA2_HAS_GENERATED_MUL
#include "generated_mul.cuh"
#endif

#if defined(_WIN32)
#define EA2_ORT_EXPORT __declspec(dllexport)
#else
#define EA2_ORT_EXPORT __attribute__((visibility("default")))
#endif

namespace approx_matmul_cuda {

constexpr int kMaxRank = 8;
constexpr int kMaxBatchRank = kMaxRank - 2;
#ifdef EA2_ENABLE_SM80_S8_TILE
constexpr int kS8ResidualTileM = 8;
constexpr int kS8ResidualTileN = 32;
constexpr int kS8ResidualTileK = 64;
#else
constexpr int kS8ResidualTileM = 16;
constexpr int kS8ResidualTileN = 16;
constexpr int kS8ResidualTileK = 32;
#endif
constexpr int64_t kCudaMaxGridX = 2147483647;
constexpr int64_t kCudaMaxGridY = 65535;

enum class MultiplierKind : int {
    kExact = 0,
    kAkbar1,
    kAkbar2,
    kAntonio,
    kHa,
    kHybrid1,
    kHybrid2,
    kMC16,
    kMomeni,
    kSabetZ,
    kAP1,
    kAP2,
    kYang1,
    kYang2,
    kYang3,
#ifdef EA2_HAS_GENERATED_MUL
    kGenerated,
#endif
};

struct TensorLayout {
    int32_t rank;
    int32_t batch_rank;
    int32_t axis;
    int32_t mat_axis0;
    int32_t mat_axis1;
    int64_t shape[kMaxRank];
    int64_t strides[kMaxRank];
    int64_t batch_dims[kMaxBatchRank];
    int64_t batch_strides[kMaxBatchRank];
};

static inline std::string Normalize(std::string value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (char ch : value) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    std::string canonical;
    canonical.reserve(lowered.size());
    for (char ch : lowered) {
        if (ch == '_' || ch == '-' || ch == ' ') {
            continue;
        }
        canonical.push_back(ch);
    }
    return canonical;
}

inline MultiplierKind ParseMultiplierKind(const std::string& name) {
    std::string canonical = Normalize(name);
    if (canonical == "kong1" || canonical == "hybrid1") canonical = "hybrid1";
    else if (canonical == "kong2" || canonical == "hybrid2") canonical = "hybrid2";
    else if (canonical == "padhy" || canonical == "mc16") canonical = "mc16";
    else if (canonical == "heamm1" || canonical == "heam1" || canonical == "heamm1b") canonical = "ap1";
    else if (canonical == "heamm2" || canonical == "heam2" || canonical == "heamm2a" || canonical == "heamm2b") canonical = "ap2";
    else if (canonical == "yangthree") canonical = "yang3";
    else if (canonical == "yangtwo") canonical = "yang2";
    else if (canonical == "yangone") canonical = "yang1";

    if (canonical == "exact") return MultiplierKind::kExact;
    if (canonical == "akbar1") return MultiplierKind::kAkbar1;
    if (canonical == "akbar2") return MultiplierKind::kAkbar2;
    if (canonical == "antonio") return MultiplierKind::kAntonio;
    if (canonical == "ha") return MultiplierKind::kHa;
    if (canonical == "hybrid1") return MultiplierKind::kHybrid1;
    if (canonical == "hybrid2") return MultiplierKind::kHybrid2;
    if (canonical == "mc16") return MultiplierKind::kMC16;
    if (canonical == "momeni") return MultiplierKind::kMomeni;
    if (canonical == "sabetz") return MultiplierKind::kSabetZ;
    if (canonical == "ap1") return MultiplierKind::kAP1;
    if (canonical == "ap2") return MultiplierKind::kAP2;
    if (canonical == "yang1") return MultiplierKind::kYang1;
    if (canonical == "yang2") return MultiplierKind::kYang2;
    if (canonical == "yang3") return MultiplierKind::kYang3;
#ifdef EA2_HAS_GENERATED_MUL
    if (canonical == "generated") return MultiplierKind::kGenerated;
#endif
    return MultiplierKind::kAkbar1;
}

static inline std::vector<float> LoadFloatBinaryFile(const std::string& path, size_t expected_count) {
    std::string clean_path = path;
    while (!clean_path.empty()) {
        const unsigned char tail = static_cast<unsigned char>(clean_path.back());
        if (tail == 0 || std::isspace(tail)) {
            clean_path.pop_back();
            continue;
        }
        break;
    }
    size_t first_non_space = 0;
    while (first_non_space < clean_path.size() && std::isspace(static_cast<unsigned char>(clean_path[first_non_space]))) {
        ++first_non_space;
    }
    if (first_non_space > 0) {
        clean_path.erase(0, first_non_space);
    }
    std::ifstream in(clean_path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw Ort::Exception(("Unable to open float32 binary: " + clean_path).c_str(), ORT_INVALID_ARGUMENT);
    }
    const std::streamoff end_pos = in.tellg();
    if (end_pos < 0) {
        throw Ort::Exception(("Unable to stat float32 binary: " + path).c_str(), ORT_INVALID_ARGUMENT);
    }
    const size_t byte_size = static_cast<size_t>(end_pos);
    const size_t expect_bytes = expected_count * sizeof(float);
    if (byte_size != expect_bytes) {
        throw Ort::Exception(
            ("Float32 binary size mismatch for " + clean_path + ": got " + std::to_string(byte_size) +
             " bytes expected " + std::to_string(expect_bytes))
                .c_str(),
            ORT_INVALID_ARGUMENT);
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> out(expected_count, 0.0f);
    if (!in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_bytes))) {
        throw Ort::Exception(("Failed reading float32 binary: " + clean_path).c_str(), ORT_INVALID_ARGUMENT);
    }
    return out;
}

static inline std::vector<int32_t> LoadInt32BinaryFile(const std::string& path, size_t expected_count) {
    std::string clean_path = path;
    while (!clean_path.empty()) {
        const unsigned char tail = static_cast<unsigned char>(clean_path.back());
        if (tail == 0 || std::isspace(tail)) {
            clean_path.pop_back();
            continue;
        }
        break;
    }
    size_t first_non_space = 0;
    while (first_non_space < clean_path.size() && std::isspace(static_cast<unsigned char>(clean_path[first_non_space]))) {
        ++first_non_space;
    }
    if (first_non_space > 0) {
        clean_path.erase(0, first_non_space);
    }
    std::ifstream in(clean_path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw Ort::Exception(("Unable to open int32 binary: " + clean_path).c_str(), ORT_INVALID_ARGUMENT);
    }
    const std::streamoff end_pos = in.tellg();
    if (end_pos < 0) {
        throw Ort::Exception(("Unable to stat int32 binary: " + path).c_str(), ORT_INVALID_ARGUMENT);
    }
    const size_t byte_size = static_cast<size_t>(end_pos);
    const size_t expect_bytes = expected_count * sizeof(int32_t);
    if (byte_size != expect_bytes) {
        throw Ort::Exception(
            ("Int32 binary size mismatch for " + clean_path + ": got " + std::to_string(byte_size) +
             " bytes expected " + std::to_string(expect_bytes))
                .c_str(),
            ORT_INVALID_ARGUMENT);
    }
    in.seekg(0, std::ios::beg);
    std::vector<int32_t> out(expected_count, 0);
    if (!in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_bytes))) {
        throw Ort::Exception(("Failed reading int32 binary: " + clean_path).c_str(), ORT_INVALID_ARGUMENT);
    }
    return out;
}

__device__ __forceinline__ uint16_t Eval(MultiplierKind kind, uint8_t a, uint8_t b) {
    switch (kind) {
        case MultiplierKind::kExact: return static_cast<uint16_t>(static_cast<uint16_t>(a) * static_cast<uint16_t>(b));
        case MultiplierKind::kAkbar1: return approx_mul_akbar1(a, b);
        case MultiplierKind::kAkbar2: return approx_mul_akbar2(a, b);
        case MultiplierKind::kAntonio: return approx_mul_antonio(a, b);
        case MultiplierKind::kHa: return approx_mul_ha(a, b);
        case MultiplierKind::kHybrid1: return approx_mul_hybrid1(a, b);
        case MultiplierKind::kHybrid2: return approx_mul_hybrid2(a, b);
        case MultiplierKind::kMC16: return approx_mul_mc16(a, b);
        case MultiplierKind::kMomeni: return approx_mul_momeni(a, b);
        case MultiplierKind::kSabetZ: return approx_mul_sabetz(a, b);
        case MultiplierKind::kAP1: return approx_mul_ap1(a, b);
        case MultiplierKind::kAP2: return approx_mul_ap2(a, b);
        case MultiplierKind::kYang1: return approx_mul_yang1(a, b);
        case MultiplierKind::kYang2: return approx_mul_yang2(a, b);
        case MultiplierKind::kYang3: return approx_mul_yang3(a, b);
#ifdef EA2_HAS_GENERATED_MUL
        case MultiplierKind::kGenerated: return approx_mul_generated(a, b);
#endif
        default: return approx_mul_akbar1(a, b);
    }
}

__device__ __forceinline__ int32_t QuantizeToQ(float value, float scale, int32_t zp, int32_t qmin, int32_t qmax) {
    double scaled = 0.0;
    if (scale != 0.0f) {
        scaled = static_cast<double>(value) / static_cast<double>(scale);
    }
    const double shifted = scaled + static_cast<double>(zp);
    long long rounded = llrint(shifted);
    if (rounded < qmin) rounded = qmin;
    if (rounded > qmax) rounded = qmax;
    return static_cast<int32_t>(rounded);
}

__device__ __forceinline__ uint8_t ClampToU8(int32_t value) {
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return static_cast<uint8_t>(value);
}

__device__ __forceinline__ void UnravelBatchIndex(
    int64_t batch_index,
    const int64_t* batch_dims,
    int32_t batch_rank,
    int64_t* batch_coords) {
    for (int32_t i = batch_rank - 1; i >= 0; --i) {
        const int64_t dim = batch_dims[i] > 0 ? batch_dims[i] : 1;
        batch_coords[i] = batch_index % dim;
        batch_index /= dim;
    }
}

__device__ __forceinline__ int64_t BatchBaseOffset(
    const TensorLayout& layout,
    const int64_t* batch_coords) {
    int64_t offset = 0;
    for (int32_t i = 0; i < layout.batch_rank; ++i) {
        offset += batch_coords[i] * layout.batch_strides[i];
    }
    return offset;
}

__device__ __forceinline__ int64_t ResolveAxisCoord(
    const TensorLayout& layout,
    const int64_t* batch_coords,
    int64_t mat_index0,
    int64_t mat_index1) {
    if (layout.axis < 0) {
        return 0;
    }
    if (layout.axis == layout.mat_axis0) {
        return mat_index0;
    }
    if (layout.axis == layout.mat_axis1) {
        return mat_index1;
    }
    const int32_t input_batch_rank = layout.rank - 2;
    if (layout.axis < input_batch_rank) {
        const int32_t batch_offset = layout.batch_rank - input_batch_rank;
        const int32_t out_axis = batch_offset + layout.axis;
        if (out_axis < 0 || out_axis >= layout.batch_rank) {
            return 0;
        }
        if (layout.shape[layout.axis] == 1) {
            return 0;
        }
        return batch_coords[out_axis];
    }
    return 0;
}

__device__ __forceinline__ float ResolveScale(
    const TensorLayout& layout,
    float scalar,
    const float* scale_arr,
    const int64_t* batch_coords,
    int64_t mat_index0,
    int64_t mat_index1) {
    if (scale_arr == nullptr || layout.axis < 0) {
        return scalar;
    }
    const int64_t idx = ResolveAxisCoord(layout, batch_coords, mat_index0, mat_index1);
    return scale_arr[idx];
}

__device__ __forceinline__ int32_t ResolveZp(
    const TensorLayout& layout,
    int32_t scalar,
    const int32_t* zp_arr,
    const int64_t* batch_coords,
    int64_t mat_index0,
    int64_t mat_index1) {
    if (zp_arr == nullptr || layout.axis < 0) {
        return scalar;
    }
    const int64_t idx = ResolveAxisCoord(layout, batch_coords, mat_index0, mat_index1);
    return zp_arr[idx];
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float u01_from_u32(uint32_t x) {
    return (static_cast<float>(x) + 1.0f) * 2.3283064365386963e-10f;
}

__device__ __forceinline__ int32_t EvalExactWithDirectCondmixNoise(
    uint8_t a,
    uint8_t b,
    const float* __restrict__ p0_grid,
    const float* __restrict__ mix_grid,
    const float* __restrict__ atom1_grid,
    const float* __restrict__ atom2_grid,
    uint64_t seed,
    uint64_t out_index,
    int32_t k_index,
    float noise_strength,
    int32_t force_nonzero_when_nonzero_branch) {
    const uint16_t m = static_cast<uint16_t>(static_cast<uint16_t>(a) * static_cast<uint16_t>(b));
    if (m == 0) return 0;
    if (!p0_grid || !mix_grid || !atom1_grid || !atom2_grid) return static_cast<int32_t>(m);

    const uint32_t idx = static_cast<uint32_t>(static_cast<uint32_t>(a) * 256u + static_cast<uint32_t>(b));
    uint64_t key = seed;
    key ^= (out_index + 0x9e3779b97f4a7c15ULL) * 0xbf58476d1ce4e5b9ULL;
    key ^= (static_cast<uint64_t>(k_index) + 17ULL) * 0x94d049bb133111ebULL;
    key ^= static_cast<uint64_t>(idx) * 0x1d8e4e27c47d124FULL;
    const uint64_t r0 = splitmix64(key);
    const uint64_t r1 = splitmix64(r0);
    const float u0 = u01_from_u32(static_cast<uint32_t>(r0 >> 32));
    const float u1 = u01_from_u32(static_cast<uint32_t>(r1 >> 32));

    const float p0 = fminf(fmaxf(p0_grid[idx], 0.0f), 1.0f);
    if (u0 < p0) return static_cast<int32_t>(m);
    const float mix = fminf(fmaxf(mix_grid[idx], 0.0f), 1.0f);
    const float atom = ((u1 < mix) ? atom2_grid[idx] : atom1_grid[idx]) * noise_strength;
    int64_t e = static_cast<int64_t>(llrintf(atom));
    if (force_nonzero_when_nonzero_branch && e == 0 && atom != 0.0f) {
        e = (atom < 0.0f) ? -1 : 1;
    }
    int64_t noisy64 = static_cast<int64_t>(m) + e;
    constexpr int64_t kInt32Min = static_cast<int64_t>(-2147483647 - 1);
    constexpr int64_t kInt32Max = static_cast<int64_t>(2147483647);
    if (noisy64 < kInt32Min) noisy64 = kInt32Min;
    if (noisy64 > kInt32Max) noisy64 = kInt32Max;
    return static_cast<int32_t>(noisy64);
}

__device__ __forceinline__ float TileDctNormDevice(int idx, int size) {
    return (idx == 0) ? sqrtf(1.0f / static_cast<float>(size)) : sqrtf(2.0f / static_cast<float>(size));
}

__device__ __forceinline__ int32_t EvalExactWithCachedSurfaceError(
    uint8_t a,
    uint8_t b,
    const int32_t* __restrict__ error_grid) {
    const uint16_t m = static_cast<uint16_t>(static_cast<uint16_t>(a) * static_cast<uint16_t>(b));
    if (m == 0) return 0;
    if (!error_grid) return static_cast<int32_t>(m);
    const uint32_t idx = static_cast<uint32_t>(static_cast<uint32_t>(a) * 256u + static_cast<uint32_t>(b));
    int64_t noisy64 = static_cast<int64_t>(m) + static_cast<int64_t>(error_grid[idx]);
    constexpr int64_t kInt32Min = static_cast<int64_t>(-2147483647 - 1);
    constexpr int64_t kInt32Max = static_cast<int64_t>(2147483647);
    if (noisy64 < kInt32Min) noisy64 = kInt32Min;
    if (noisy64 > kInt32Max) noisy64 = kInt32Max;
    return static_cast<int32_t>(noisy64);
}

__global__ void decode_oracle_surface_error_grid_kernel(
    const float* __restrict__ mean_surface,
    const float* __restrict__ basis,
    const float* __restrict__ global_coeff,
    int32_t basis_rank,
    const float* __restrict__ local_coeff,
    int32_t residual_tiles,
    int32_t residual_freq_h,
    int32_t residual_freq_w,
    float noise_strength,
    int32_t force_nonzero,
    int32_t* __restrict__ error_grid) {
    constexpr int32_t kSide = 256;
    constexpr int32_t kTableSize = kSide * kSide;
    const int64_t idx = blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    if (idx >= kTableSize) return;

    const int32_t a = static_cast<int32_t>(idx / kSide);
    const int32_t b = static_cast<int32_t>(idx % kSide);
    if (a == 0 || b == 0) {
        error_grid[idx] = 0;
        return;
    }

    float acc = mean_surface[idx];
    for (int32_t basis_idx = 0; basis_idx < basis_rank; ++basis_idx) {
        acc += global_coeff[basis_idx] * basis[static_cast<int64_t>(basis_idx) * kTableSize + idx];
    }

    const int32_t cell = kSide / residual_tiles;
    const int32_t tile_a = a / cell;
    const int32_t tile_b = b / cell;
    const int32_t local_a = a % cell;
    const int32_t local_b = b % cell;
    const int32_t tile_offset = ((tile_a * residual_tiles) + tile_b) * residual_freq_h * residual_freq_w;
    constexpr float kPi = 3.14159265358979323846f;
    for (int32_t fy = 0; fy < residual_freq_h; ++fy) {
        const float basis_y = TileDctNormDevice(fy, cell)
            * cosf(kPi * (static_cast<float>(local_a) + 0.5f) * static_cast<float>(fy) / static_cast<float>(cell));
        for (int32_t fx = 0; fx < residual_freq_w; ++fx) {
            const float basis_x = TileDctNormDevice(fx, cell)
                * cosf(kPi * (static_cast<float>(local_b) + 0.5f) * static_cast<float>(fx) / static_cast<float>(cell));
            const int32_t coeff_idx = tile_offset + fy * residual_freq_w + fx;
            acc += local_coeff[coeff_idx] * basis_y * basis_x;
        }
    }

    const float scaled = acc * noise_strength;
    int64_t err_i64 = static_cast<int64_t>(llrintf(scaled));
    if (force_nonzero != 0 && err_i64 == 0 && scaled != 0.0f) {
        err_i64 = (scaled < 0.0f) ? -1 : 1;
    }
    constexpr int64_t kInt32Min = static_cast<int64_t>(-2147483647 - 1);
    constexpr int64_t kInt32Max = static_cast<int64_t>(2147483647);
    if (err_i64 < kInt32Min) err_i64 = kInt32Min;
    if (err_i64 > kInt32Max) err_i64 = kInt32Max;
    error_grid[idx] = static_cast<int32_t>(err_i64);
}

__global__ void approx_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t batch_total,
    int64_t m_dim,
    int64_t k_dim,
    int64_t n_dim,
    TensorLayout a_layout,
    TensorLayout b_layout,
    float scale_a_scalar,
    const float* __restrict__ scale_a_arr,
    int32_t zp_a_scalar,
    const int32_t* __restrict__ zp_a_arr,
    int32_t qmin_a,
    int32_t qmax_a,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    int32_t zp_b_scalar,
    const int32_t* __restrict__ zp_b_arr,
    int32_t qmin_b,
    int32_t qmax_b,
    float alpha,
    MultiplierKind kind,
    const float* __restrict__ noise_p0_grid,
    const float* __restrict__ noise_mix_grid,
    const float* __restrict__ noise_atom1_grid,
    const float* __restrict__ noise_atom2_grid,
    const int32_t* __restrict__ noise_oracle_error_grid,
    const int32_t* __restrict__ lut_delta_table,
    uint64_t noise_seed,
    float noise_strength,
    int32_t noise_force_nonzero,
    int32_t noise_direct_condmix,
    int32_t noise_oracle_surface_coeff_cached,
    int32_t use_lut_table,
    int32_t add_to_base) {
    const int64_t total = batch_total * m_dim * n_dim;
    const int64_t linear_index = blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    if (linear_index >= total) {
        return;
    }

    int64_t tmp = linear_index;
    const int64_t n = tmp % n_dim;
    tmp /= n_dim;
    const int64_t m = tmp % m_dim;
    tmp /= m_dim;
    const int64_t batch_index = tmp;

    int64_t batch_coords[kMaxBatchRank] = {};
    UnravelBatchIndex(batch_index, a_layout.batch_dims, a_layout.batch_rank, batch_coords);

    const int64_t a_batch_base = BatchBaseOffset(a_layout, batch_coords);
    const int64_t b_batch_base = BatchBaseOffset(b_layout, batch_coords);

    float acc = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
        const int64_t a_offset =
            a_batch_base +
            m * a_layout.strides[a_layout.mat_axis0] +
            k * a_layout.strides[a_layout.mat_axis1];
        const int64_t b_offset =
            b_batch_base +
            k * b_layout.strides[b_layout.mat_axis0] +
            n * b_layout.strides[b_layout.mat_axis1];

        const float a_value = a[a_offset];
        const float b_value = b[b_offset];

        // Keep the legacy full-recompute behavior for non-LUT users.  The
        // residual LUT path intentionally includes zero-point codes because
        // its error table is defined over the raw unsigned multiplier inputs.
        if (add_to_base == 0 && (a_value == 0.0f || b_value == 0.0f)) {
            continue;
        }

        const float scale_a = ResolveScale(a_layout, scale_a_scalar, scale_a_arr, batch_coords, m, k);
        const int32_t zp_a = ResolveZp(a_layout, zp_a_scalar, zp_a_arr, batch_coords, m, k);
        const float scale_b = ResolveScale(b_layout, scale_b_scalar, scale_b_arr, batch_coords, k, n);
        const int32_t zp_b = ResolveZp(b_layout, zp_b_scalar, zp_b_arr, batch_coords, k, n);

        const int32_t qa = QuantizeToQ(a_value, scale_a, zp_a, qmin_a, qmax_a);
        const int32_t qb = QuantizeToQ(b_value, scale_b, zp_b, qmin_b, qmax_b);
        const int32_t ua = qa - qmin_a;
        const int32_t ub = qb - qmin_b;
        const uint8_t ua8 = ClampToU8(ua);
        const uint8_t ub8 = ClampToU8(ub);
        const uint32_t lut_idx = static_cast<uint32_t>(static_cast<uint32_t>(ua8) * 256u + static_cast<uint32_t>(ub8));
        const int32_t exact_unsigned = static_cast<int32_t>(static_cast<uint16_t>(ua8) * static_cast<uint16_t>(ub8));
        const int32_t approx_unsigned = (use_lut_table != 0)
            ? (exact_unsigned + (lut_delta_table ? lut_delta_table[lut_idx] : 0))
            : ((noise_oracle_surface_coeff_cached != 0)
                   ? EvalExactWithCachedSurfaceError(ua8, ub8, noise_oracle_error_grid)
                   : ((noise_direct_condmix != 0)
                          ? EvalExactWithDirectCondmixNoise(
                                ua8,
                                ub8,
                                noise_p0_grid,
                                noise_mix_grid,
                                noise_atom1_grid,
                                noise_atom2_grid,
                                noise_seed,
                                static_cast<uint64_t>(linear_index),
                                static_cast<int32_t>(k),
                                noise_strength,
                                noise_force_nonzero)
                          : static_cast<int32_t>(Eval(kind, ua8, ub8))));
        const int32_t term = (add_to_base != 0)
            ? (approx_unsigned - exact_unsigned)
            : (approx_unsigned +
               ua * (qmin_b - zp_b) +
               ub * (qmin_a - zp_a) +
               (qmin_a - zp_a) * (qmin_b - zp_b));
        acc += static_cast<float>(term) * (scale_a * scale_b);
    }

    y[linear_index] = (add_to_base != 0 ? exact_base[linear_index] : 0.0f) + acc * alpha;
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void approx_matmul_tiled_fast_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t batch_total,
    int64_t m_dim,
    int64_t k_dim,
    int64_t n_dim,
    TensorLayout a_layout,
    TensorLayout b_layout,
    float scale_a_scalar,
    int32_t zp_a_scalar,
    int32_t qmin_a,
    int32_t qmax_a,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    int32_t zp_b_scalar,
    const int32_t* __restrict__ zp_b_arr,
    int32_t qmin_b,
    int32_t qmax_b,
    float alpha,
    MultiplierKind kind,
    const int32_t* __restrict__ lut_delta_table,
    int32_t use_lut_table,
    int32_t add_to_base) {
    __shared__ uint8_t sh_a_u[TILE_M][TILE_K];
    __shared__ uint8_t sh_b_u[TILE_K][TILE_N];
    __shared__ uint8_t sh_a_zero[TILE_M][TILE_K];
    __shared__ uint8_t sh_b_zero[TILE_K][TILE_N];

    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int64_t batch_index = static_cast<int64_t>(blockIdx.z);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m = m0 + ty;
    const int64_t n = n0 + tx;

    int64_t batch_coords[kMaxBatchRank] = {};
    UnravelBatchIndex(batch_index, a_layout.batch_dims, a_layout.batch_rank, batch_coords);
    const int64_t a_batch_base = BatchBaseOffset(a_layout, batch_coords);
    const int64_t b_batch_base = BatchBaseOffset(b_layout, batch_coords);

    const float scale_a = scale_a_scalar;
    const int32_t corr_a = qmin_a - zp_a_scalar;
    const float scale_b = (n < n_dim)
        ? ((scale_b_arr != nullptr) ? scale_b_arr[n] : scale_b_scalar)
        : scale_b_scalar;
    const int32_t zp_b = (n < n_dim)
        ? ((zp_b_arr != nullptr) ? zp_b_arr[n] : zp_b_scalar)
        : zp_b_scalar;
    const int32_t corr_b = qmin_b - zp_b;
    const int32_t corr_const = corr_a * corr_b;

    int32_t acc_i = 0;

    for (int64_t k0 = 0; k0 < k_dim; k0 += TILE_K) {
        for (int32_t idx = ty * TILE_N + tx; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
            const int32_t tile_m = idx / TILE_K;
            const int32_t tile_k = idx % TILE_K;
            const int64_t gm = m0 + tile_m;
            const int64_t gk = k0 + tile_k;
            if (gm < m_dim && gk < k_dim) {
                const int64_t a_offset =
                    a_batch_base +
                    gm * a_layout.strides[a_layout.mat_axis0] +
                    gk * a_layout.strides[a_layout.mat_axis1];
                const float a_value = a[a_offset];
                sh_a_zero[tile_m][tile_k] = (a_value == 0.0f) ? 1 : 0;
                const int32_t qa = QuantizeToQ(a_value, scale_a, zp_a_scalar, qmin_a, qmax_a);
                sh_a_u[tile_m][tile_k] = ClampToU8(qa - qmin_a);
            } else {
                sh_a_zero[tile_m][tile_k] = 1;
                sh_a_u[tile_m][tile_k] = 0;
            }
        }

        for (int32_t idx = ty * TILE_N + tx; idx < TILE_K * TILE_N; idx += TILE_M * TILE_N) {
            const int32_t tile_k = idx / TILE_N;
            const int32_t tile_n = idx % TILE_N;
            const int64_t gk = k0 + tile_k;
            const int64_t gn = n0 + tile_n;
            if (gk < k_dim && gn < n_dim) {
                const int64_t b_offset =
                    b_batch_base +
                    gk * b_layout.strides[b_layout.mat_axis0] +
                    gn * b_layout.strides[b_layout.mat_axis1];
                const float b_value = b[b_offset];
                sh_b_zero[tile_k][tile_n] = (b_value == 0.0f) ? 1 : 0;
                const float scale_b_load = (scale_b_arr != nullptr) ? scale_b_arr[gn] : scale_b_scalar;
                const int32_t zp_b_load = (zp_b_arr != nullptr) ? zp_b_arr[gn] : zp_b_scalar;
                const int32_t qb = QuantizeToQ(b_value, scale_b_load, zp_b_load, qmin_b, qmax_b);
                sh_b_u[tile_k][tile_n] = ClampToU8(qb - qmin_b);
            } else {
                sh_b_zero[tile_k][tile_n] = 1;
                sh_b_u[tile_k][tile_n] = 0;
            }
        }
        __syncthreads();

        if (m < m_dim && n < n_dim) {
            const int64_t remaining_k = k_dim - k0;
            const int32_t k_limit = static_cast<int32_t>((remaining_k < TILE_K) ? remaining_k : TILE_K);
            for (int32_t kk = 0; kk < k_limit; ++kk) {
                if (add_to_base == 0 && (sh_a_zero[ty][kk] || sh_b_zero[kk][tx])) {
                    continue;
                }
                const uint8_t ua8 = sh_a_u[ty][kk];
                const uint8_t ub8 = sh_b_u[kk][tx];
                const uint32_t lut_idx = static_cast<uint32_t>(static_cast<uint32_t>(ua8) * 256u + static_cast<uint32_t>(ub8));
                const int32_t exact_unsigned = static_cast<int32_t>(static_cast<uint16_t>(ua8) * static_cast<uint16_t>(ub8));
                const int32_t approx_unsigned = (use_lut_table != 0)
                    ? (exact_unsigned + (lut_delta_table ? lut_delta_table[lut_idx] : 0))
                    : static_cast<int32_t>(Eval(kind, ua8, ub8));
                const int32_t term = (add_to_base != 0)
                    ? (approx_unsigned - exact_unsigned)
                    : (approx_unsigned + static_cast<int32_t>(ua8) * corr_b + static_cast<int32_t>(ub8) * corr_a + corr_const);
                acc_i += term;
            }
        }
        __syncthreads();
    }

    if (m < m_dim && n < n_dim) {
        const int64_t out_index = (batch_index * m_dim + m) * n_dim + n;
        y[out_index] = (add_to_base != 0 ? exact_base[out_index] : 0.0f) +
            static_cast<float>(acc_i) * (scale_a * scale_b) * alpha;
    }
}

// Residual-LUT fast path for legacy U8 QDQ graphs. The graph passes the
// original U8 code tensors, so this avoids redoing float division, rounding,
// and clamping for every term in the K reduction.
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void approx_matmul_tiled_u8_residual_kernel(
    const uint8_t* __restrict__ a_q,
    const uint8_t* __restrict__ b_q,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t batch_total,
    int64_t m_dim,
    int64_t k_dim,
    int64_t n_dim,
    TensorLayout a_layout,
    TensorLayout b_layout,
    int32_t qmin_a,
    int32_t qmin_b,
    float scale_a,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    float alpha,
    MultiplierKind kind,
    const int32_t* __restrict__ lut_delta_table,
    int32_t use_lut_table) {
    __shared__ uint8_t sh_a_u[TILE_M][TILE_K];
    __shared__ uint8_t sh_b_u[TILE_K][TILE_N];

    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int64_t batch_index = static_cast<int64_t>(blockIdx.z);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m = m0 + ty;
    const int64_t n = n0 + tx;

    int64_t batch_coords[kMaxBatchRank] = {};
    UnravelBatchIndex(batch_index, a_layout.batch_dims, a_layout.batch_rank, batch_coords);
    const int64_t a_batch_base = BatchBaseOffset(a_layout, batch_coords);
    const int64_t b_batch_base = BatchBaseOffset(b_layout, batch_coords);
    const float scale_b = (n < n_dim)
        ? ((scale_b_arr != nullptr) ? scale_b_arr[n] : scale_b_scalar)
        : scale_b_scalar;
    int32_t residual_acc = 0;

    for (int64_t k0 = 0; k0 < k_dim; k0 += TILE_K) {
        for (int32_t idx = ty * TILE_N + tx; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
            const int32_t tile_m = idx / TILE_K;
            const int32_t tile_k = idx % TILE_K;
            const int64_t gm = m0 + tile_m;
            const int64_t gk = k0 + tile_k;
            if (gm < m_dim && gk < k_dim) {
                const int64_t a_offset =
                    a_batch_base +
                    gm * a_layout.strides[a_layout.mat_axis0] +
                    gk * a_layout.strides[a_layout.mat_axis1];
                sh_a_u[tile_m][tile_k] = ClampToU8(static_cast<int32_t>(a_q[a_offset]) - qmin_a);
            } else {
                sh_a_u[tile_m][tile_k] = 0;
            }
        }

        for (int32_t idx = ty * TILE_N + tx; idx < TILE_K * TILE_N; idx += TILE_M * TILE_N) {
            const int32_t tile_k = idx / TILE_N;
            const int32_t tile_n = idx % TILE_N;
            const int64_t gk = k0 + tile_k;
            const int64_t gn = n0 + tile_n;
            if (gk < k_dim && gn < n_dim) {
                const int64_t b_offset =
                    b_batch_base +
                    gk * b_layout.strides[b_layout.mat_axis0] +
                    gn * b_layout.strides[b_layout.mat_axis1];
                sh_b_u[tile_k][tile_n] = ClampToU8(static_cast<int32_t>(b_q[b_offset]) - qmin_b);
            } else {
                sh_b_u[tile_k][tile_n] = 0;
            }
        }
        __syncthreads();

        if (m < m_dim && n < n_dim) {
            const int64_t remaining_k = k_dim - k0;
            const int32_t k_limit = static_cast<int32_t>((remaining_k < TILE_K) ? remaining_k : TILE_K);
            for (int32_t kk = 0; kk < k_limit; ++kk) {
                const uint8_t ua8 = sh_a_u[ty][kk];
                const uint8_t ub8 = sh_b_u[kk][tx];
                const uint32_t lut_idx = static_cast<uint32_t>(static_cast<uint32_t>(ua8) * 256u + static_cast<uint32_t>(ub8));
                const int32_t exact_unsigned = static_cast<int32_t>(static_cast<uint16_t>(ua8) * static_cast<uint16_t>(ub8));
                const int32_t residual = (use_lut_table != 0)
                    ? (lut_delta_table ? lut_delta_table[lut_idx] : 0)
                    : (static_cast<int32_t>(Eval(kind, ua8, ub8)) - exact_unsigned);
                residual_acc += residual;
            }
        }
        __syncthreads();
    }

    if (m < m_dim && n < n_dim) {
        const int64_t out_index = (batch_index * m_dim + m) * n_dim + n;
        y[out_index] = exact_base[out_index] +
            static_cast<float>(residual_acc) * (scale_a * scale_b) * alpha;
    }
}

// S8S8 sign-magnitude residual path. The LUT is indexed by raw signed QDQ
// code plus 128 and stores T_s8(q_a, q_b) - q_a * q_b. The exact MatMul input
// already contains affine zero-point correction. A Q/DQ semantic zero
// (q == zp) must not receive an approximate residual, preserving 0 * x = 0
// without changing the multiplier LUT for nonzero operands.
__global__ void approx_matmul_s8_residual_kernel(
    const int8_t* __restrict__ a_q,
    const int8_t* __restrict__ b_q,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t batch_total,
    int64_t m_dim,
    int64_t k_dim,
    int64_t n_dim,
    TensorLayout a_layout,
    TensorLayout b_layout,
    float scale_a_scalar,
    const float* __restrict__ scale_a_arr,
    int32_t zp_a_scalar,
    const int32_t* __restrict__ zp_a_arr,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    int32_t zp_b_scalar,
    const int32_t* __restrict__ zp_b_arr,
    float alpha,
    const int32_t* __restrict__ lut_delta_table) {
    const int64_t total = batch_total * m_dim * n_dim;
    const int64_t linear_index = blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    if (linear_index >= total) {
        return;
    }

    int64_t tmp = linear_index;
    const int64_t n = tmp % n_dim;
    tmp /= n_dim;
    const int64_t m = tmp % m_dim;
    tmp /= m_dim;
    const int64_t batch_index = tmp;

    int64_t batch_coords[kMaxBatchRank] = {};
    UnravelBatchIndex(batch_index, a_layout.batch_dims, a_layout.batch_rank, batch_coords);
    const int64_t a_batch_base = BatchBaseOffset(a_layout, batch_coords);
    const int64_t b_batch_base = BatchBaseOffset(b_layout, batch_coords);

    float residual = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
        const int64_t a_offset =
            a_batch_base +
            m * a_layout.strides[a_layout.mat_axis0] +
            k * a_layout.strides[a_layout.mat_axis1];
        const int64_t b_offset =
            b_batch_base +
            k * b_layout.strides[b_layout.mat_axis0] +
            n * b_layout.strides[b_layout.mat_axis1];
        const int32_t qa = static_cast<int32_t>(a_q[a_offset]);
        const int32_t qb = static_cast<int32_t>(b_q[b_offset]);
        const int32_t zp_a = ResolveZp(a_layout, zp_a_scalar, zp_a_arr, batch_coords, m, k);
        const int32_t zp_b = ResolveZp(b_layout, zp_b_scalar, zp_b_arr, batch_coords, k, n);
        if (qa == zp_a || qb == zp_b) {
            continue;
        }
        const uint32_t lut_idx = static_cast<uint32_t>((qa + 128) * 256 + (qb + 128));
        const float scale_a = ResolveScale(a_layout, scale_a_scalar, scale_a_arr, batch_coords, m, k);
        const float scale_b = ResolveScale(b_layout, scale_b_scalar, scale_b_arr, batch_coords, k, n);
        residual += static_cast<float>(lut_delta_table[lut_idx]) * (scale_a * scale_b);
    }

    y[linear_index] = exact_base[linear_index] + residual * alpha;
}

template <typename ResidualAccumulator, int TILE_M, int TILE_N, int TILE_K>
__global__ void approx_matmul_tiled_s8_residual_kernel(
    const int8_t* __restrict__ a_q,
    const int8_t* __restrict__ b_q,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t batch_total,
    int64_t m_dim,
    int64_t k_dim,
    int64_t n_dim,
    TensorLayout a_layout,
    TensorLayout b_layout,
    float scale_a,
    int32_t zp_a,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    int32_t zp_b_scalar,
    const int32_t* __restrict__ zp_b_arr,
    float alpha,
    const int32_t* __restrict__ lut_delta_table) {
    __shared__ int8_t sh_a_q[TILE_M][TILE_K];
    __shared__ int8_t sh_b_q[TILE_K][TILE_N];

    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int64_t batch_index = static_cast<int64_t>(blockIdx.z);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m = m0 + ty;
    const int64_t n = n0 + tx;

    int64_t batch_coords[kMaxBatchRank] = {};
    UnravelBatchIndex(batch_index, a_layout.batch_dims, a_layout.batch_rank, batch_coords);
    const int64_t a_batch_base = BatchBaseOffset(a_layout, batch_coords);
    const int64_t b_batch_base = BatchBaseOffset(b_layout, batch_coords);
    const float scale_b = (n < n_dim)
        ? ((scale_b_arr != nullptr) ? scale_b_arr[n] : scale_b_scalar)
        : scale_b_scalar;
    const int32_t zp_b = (n < n_dim)
        ? ((zp_b_arr != nullptr) ? zp_b_arr[n] : zp_b_scalar)
        : zp_b_scalar;
    ResidualAccumulator residual_acc = 0;

    for (int64_t k0 = 0; k0 < k_dim; k0 += TILE_K) {
        for (int32_t idx = ty * TILE_N + tx; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
            const int32_t tile_m = idx / TILE_K;
            const int32_t tile_k = idx % TILE_K;
            const int64_t gm = m0 + tile_m;
            const int64_t gk = k0 + tile_k;
            if (gm < m_dim && gk < k_dim) {
                const int64_t a_offset =
                    a_batch_base +
                    gm * a_layout.strides[a_layout.mat_axis0] +
                    gk * a_layout.strides[a_layout.mat_axis1];
                sh_a_q[tile_m][tile_k] = a_q[a_offset];
            } else {
                sh_a_q[tile_m][tile_k] = 0;
            }
        }

        for (int32_t idx = ty * TILE_N + tx; idx < TILE_K * TILE_N; idx += TILE_M * TILE_N) {
            const int32_t tile_k = idx / TILE_N;
            const int32_t tile_n = idx % TILE_N;
            const int64_t gk = k0 + tile_k;
            const int64_t gn = n0 + tile_n;
            if (gk < k_dim && gn < n_dim) {
                const int64_t b_offset =
                    b_batch_base +
                    gk * b_layout.strides[b_layout.mat_axis0] +
                    gn * b_layout.strides[b_layout.mat_axis1];
                sh_b_q[tile_k][tile_n] = b_q[b_offset];
            } else {
                sh_b_q[tile_k][tile_n] = 0;
            }
        }
        __syncthreads();

        if (m < m_dim && n < n_dim) {
            const int64_t remaining_k = k_dim - k0;
            const int32_t k_limit = static_cast<int32_t>((remaining_k < TILE_K) ? remaining_k : TILE_K);
            for (int32_t kk = 0; kk < k_limit; ++kk) {
                const int32_t qa = static_cast<int32_t>(sh_a_q[ty][kk]);
                const int32_t qb = static_cast<int32_t>(sh_b_q[kk][tx]);
                if (qa == zp_a || qb == zp_b) {
                    continue;
                }
                const uint32_t lut_idx = static_cast<uint32_t>((qa + 128) * 256 + (qb + 128));
                residual_acc += static_cast<ResidualAccumulator>(lut_delta_table[lut_idx]);
            }
        }
        __syncthreads();
    }

    if (m < m_dim && n < n_dim) {
        const int64_t out_index = (batch_index * m_dim + m) * n_dim + n;
        y[out_index] = exact_base[out_index] +
            static_cast<float>(residual_acc) * (scale_a * scale_b) * alpha;
    }
}

// This specialization is deliberately narrow.  A static RHS linear MatMul
// has A=[B, M, K], B=[K, N], and an output/base that is contiguous [B, M, N].
// Flattening B and M removes repeated batch-coordinate unraveling and lets
// adjacent blocks walk one contiguous A/output plane.  Keep the same K order
// and one-thread-per-output accumulator as the general tiled kernel so the
// residual result is bitwise identical for the same inputs.
template <typename ResidualAccumulator, int TILE_M, int TILE_N, int TILE_K>
__global__ void approx_matmul_flattened_static_rhs_s8_residual_kernel(
    const int8_t* __restrict__ a_q,
    const int8_t* __restrict__ b_q,
    const float* __restrict__ exact_base,
    float* __restrict__ y,
    int64_t flat_m_total,
    int64_t k_dim,
    int64_t n_dim,
    float scale_a,
    int32_t zp_a,
    float scale_b_scalar,
    const float* __restrict__ scale_b_arr,
    int32_t zp_b_scalar,
    const int32_t* __restrict__ zp_b_arr,
    float alpha,
    const int32_t* __restrict__ lut_delta_table) {
    __shared__ int8_t sh_a_q[TILE_M][TILE_K];
    __shared__ int8_t sh_b_q[TILE_K][TILE_N];

    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int64_t flat_m0 = static_cast<int64_t>(blockIdx.x) * TILE_M;
    const int64_t n0 = static_cast<int64_t>(blockIdx.y) * TILE_N;
    const int64_t flat_m = flat_m0 + ty;
    const int64_t n = n0 + tx;
    const float scale_b = (n < n_dim)
        ? ((scale_b_arr != nullptr) ? scale_b_arr[n] : scale_b_scalar)
        : scale_b_scalar;
    const int32_t zp_b = (n < n_dim)
        ? ((zp_b_arr != nullptr) ? zp_b_arr[n] : zp_b_scalar)
        : zp_b_scalar;
    ResidualAccumulator residual_acc = 0;

    for (int64_t k0 = 0; k0 < k_dim; k0 += TILE_K) {
        for (int32_t idx = ty * TILE_N + tx; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
            const int32_t tile_m = idx / TILE_K;
            const int32_t tile_k = idx % TILE_K;
            const int64_t gm = flat_m0 + tile_m;
            const int64_t gk = k0 + tile_k;
            if (gm < flat_m_total && gk < k_dim) {
                sh_a_q[tile_m][tile_k] = a_q[gm * k_dim + gk];
            } else {
                sh_a_q[tile_m][tile_k] = 0;
            }
        }

        for (int32_t idx = ty * TILE_N + tx; idx < TILE_K * TILE_N; idx += TILE_M * TILE_N) {
            const int32_t tile_k = idx / TILE_N;
            const int32_t tile_n = idx % TILE_N;
            const int64_t gk = k0 + tile_k;
            const int64_t gn = n0 + tile_n;
            if (gk < k_dim && gn < n_dim) {
                sh_b_q[tile_k][tile_n] = b_q[gk * n_dim + gn];
            } else {
                sh_b_q[tile_k][tile_n] = 0;
            }
        }
        __syncthreads();

        if (flat_m < flat_m_total && n < n_dim) {
            const int64_t remaining_k = k_dim - k0;
            const int32_t k_limit = static_cast<int32_t>((remaining_k < TILE_K) ? remaining_k : TILE_K);
            for (int32_t kk = 0; kk < k_limit; ++kk) {
                const int32_t qa = static_cast<int32_t>(sh_a_q[ty][kk]);
                const int32_t qb = static_cast<int32_t>(sh_b_q[kk][tx]);
                if (qa == zp_a || qb == zp_b) {
                    continue;
                }
                const uint32_t lut_idx = static_cast<uint32_t>((qa + 128) * 256 + (qb + 128));
                residual_acc += static_cast<ResidualAccumulator>(lut_delta_table[lut_idx]);
            }
        }
        __syncthreads();
    }

    if (flat_m < flat_m_total && n < n_dim) {
        const int64_t out_index = flat_m * n_dim + n;
        y[out_index] = exact_base[out_index] +
            static_cast<float>(residual_acc) * (scale_a * scale_b) * alpha;
    }
}

class ApproxMatMulCudaKernel {
public:
    ApproxMatMulCudaKernel(const OrtApi& api, const OrtKernelInfo* info, bool signed_code_lut = false)
        : api_(&api), signed_code_lut_(signed_code_lut) {
        scale_a_ = GetFloatAttributeAlias(info, {"scale_a", "scale_x"}, 1.0f);
        scale_b_ = GetFloatAttributeAlias(info, {"scale_b", "scale_w"}, 1.0f);

        zp_a_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"zp_a", "zp_x"}, 0));
        zp_b_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"zp_b", "zp_w"}, 0));
        qmin_a_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"qmin_a", "qmin_x"}, 0));
        qmax_a_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"qmax_a", "qmax_x"}, 255));
        qmin_b_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"qmin_b", "qmin_w"}, 0));
        qmax_b_ = static_cast<int32_t>(GetInt64AttributeAlias(info, {"qmax_b", "qmax_w"}, 255));

        raw_axis_a_ = GetOptionalInt64AttributeAlias(info, {"axis_a", "axis_x"});
        raw_axis_b_ = GetOptionalInt64AttributeAlias(info, {"axis_b", "axis_w"});
        trans_a_ = GetBoolAttributeAlias(info, {"trans_a", "transA"}, false);
        trans_b_ = GetBoolAttributeAlias(info, {"trans_b", "transB"}, false);
        alpha_ = GetFloatAttributeOr(info, "alpha", 1.0f);
        base_input_ = (GetInt64AttributeOr(info, "base_input", 0) != 0);
        if (signed_code_lut_ && !base_input_) {
            throw Ort::Exception("ApproxMatMulCudaS8Lut requires base_input=1", ORT_INVALID_ARGUMENT);
        }
        if (signed_code_lut_ && GetStringAttributeOr(info, "lut_domain", "") != "s8s8_sign_magnitude") {
            throw Ort::Exception(
                "ApproxMatMulCudaS8Lut requires lut_domain=s8s8_sign_magnitude",
                ORT_INVALID_ARGUMENT);
        }
        if (signed_code_lut_ && GetInt64AttributeOr(info, "semantic_zero_mask", 0) != 1) {
            throw Ort::Exception(
                "ApproxMatMulCudaS8Lut requires semantic_zero_mask=1",
                ORT_INVALID_ARGUMENT);
        }
        s8_residual_abs_bound_ = GetInt64AttributeOr(info, "s8_residual_abs_bound", -1);

        scale_a_arr_ = GetFloatsAttributeAlias(info, {"scale_a_arr", "scale_x_arr"}, {});
        scale_b_arr_ = GetFloatsAttributeAlias(info, {"scale_b_arr", "scale_w_arr"}, {});
        {
            auto zp_a_i64 = GetIntsAttributeAlias(info, {"zp_a_arr", "zp_x_arr"}, {});
            zp_a_arr_.reserve(zp_a_i64.size());
            for (auto value : zp_a_i64) {
                zp_a_arr_.push_back(static_cast<int32_t>(value));
            }
        }
        {
            auto zp_b_i64 = GetIntsAttributeAlias(info, {"zp_b_arr", "zp_w_arr"}, {});
            zp_b_arr_.reserve(zp_b_i64.size());
            for (auto value : zp_b_i64) {
                zp_b_arr_.push_back(static_cast<int32_t>(value));
            }
        }

        std::string multiplier_kind = GetStringAttributeOr(info, "multiplier_kind", "");
        if (multiplier_kind.empty()) {
            multiplier_kind = GetStringAttributeOr(info, "multiplier", "akbar1");
        }
        kind_ = ParseMultiplierKind(multiplier_kind);

        noise_enable_ = (GetInt64AttributeOr(info, "noise_enable", 0) != 0);
        noise_seed_ = static_cast<uint64_t>(GetInt64AttributeOr(info, "noise_seed", 7));
        noise_strength_ = GetFloatAttributeOr(info, "noise_strength", 1.0f);
        noise_force_nonzero_ = (GetInt64AttributeOr(info, "noise_force_nonzero", 1) != 0);
        noise_direct_condmix_ = (GetInt64AttributeOr(info, "noise_direct_condmix", 0) != 0);
        noise_oracle_surface_coeff_cached_ = (GetInt64AttributeOr(info, "noise_oracle_surface_coeff_cached", 0) != 0);
        noise_oracle_basis_mean_path_ = GetStringAttributeOr(info, "noise_oracle_basis_mean_path", "");
        noise_oracle_basis_path_ = GetStringAttributeOr(info, "noise_oracle_basis_path", "");
        noise_oracle_basis_side_ = static_cast<int32_t>(GetInt64AttributeOr(info, "noise_oracle_basis_side", 256));
        noise_oracle_basis_rank_ = static_cast<int32_t>(GetInt64AttributeOr(info, "noise_oracle_basis_rank", 0));
        noise_oracle_global_coeff_ = GetFloatsAttributeOr(info, "noise_oracle_global_coeff", {});
        noise_oracle_residual_tiles_ = static_cast<int32_t>(GetInt64AttributeOr(info, "noise_oracle_residual_tiles", 0));
        noise_oracle_residual_freq_h_ = static_cast<int32_t>(GetInt64AttributeOr(info, "noise_oracle_residual_freq_h", 0));
        noise_oracle_residual_freq_w_ = static_cast<int32_t>(GetInt64AttributeOr(info, "noise_oracle_residual_freq_w", 0));
        noise_oracle_local_coeff_ = GetFloatsAttributeOr(info, "noise_oracle_local_coeff", {});
        noise_p0_grid_ = GetFloatsAttributeOr(info, "noise_p0_grid", {});
        noise_mix_grid_ = GetFloatsAttributeOr(info, "noise_mix_grid", {});
        noise_atom1_grid_ = GetFloatsAttributeOr(info, "noise_atom1_grid", {});
        noise_atom2_grid_ = GetFloatsAttributeOr(info, "noise_atom2_grid", {});
        lut_delta_path_ = GetStringAttributeOr(info, "lut_delta_path", "");
        if (!lut_delta_path_.empty()) {
            constexpr size_t kTableSize = 256u * 256u;
            lut_delta_host_ = LoadInt32BinaryFile(lut_delta_path_, kTableSize);
            use_lut_table_ = true;
        }
        const int32_t active_noise_families =
            (noise_direct_condmix_ ? 1 : 0) +
            (noise_oracle_surface_coeff_cached_ ? 1 : 0);
        if (active_noise_families > 1) {
            throw Ort::Exception(
                "noise_direct_condmix and noise_oracle_surface_coeff_cached are mutually exclusive",
                ORT_INVALID_ARGUMENT);
        }
        if (signed_code_lut_ && (noise_enable_ || active_noise_families != 0)) {
            throw Ort::Exception(
                "ApproxMatMulCudaS8Lut only supports explicit S8S8 LUT residuals, not legacy U8 noise attributes",
                ORT_INVALID_ARGUMENT);
        }
        if (noise_enable_ && noise_direct_condmix_) {
            const size_t expect = 256u * 256u;
            if (
                noise_p0_grid_.size() != expect ||
                noise_mix_grid_.size() != expect ||
                noise_atom1_grid_.size() != expect ||
                    noise_atom2_grid_.size() != expect) {
                throw Ort::Exception("Direct condmix noise attrs must all have size 65536", ORT_INVALID_ARGUMENT);
            }
        }
        if (noise_enable_ && noise_oracle_surface_coeff_cached_) {
            constexpr size_t kTableSize = 256u * 256u;
            if (noise_oracle_basis_side_ != 256) {
                throw Ort::Exception("Oracle coeff cached noise currently requires basis_side=256", ORT_INVALID_ARGUMENT);
            }
            if (noise_oracle_basis_rank_ <= 0) {
                throw Ort::Exception("Oracle coeff cached noise requires positive basis_rank", ORT_INVALID_ARGUMENT);
            }
            if (noise_oracle_residual_tiles_ <= 0 || noise_oracle_residual_freq_h_ <= 0 || noise_oracle_residual_freq_w_ <= 0) {
                throw Ort::Exception("Oracle coeff cached noise requires positive residual tile/freq attrs", ORT_INVALID_ARGUMENT);
            }
            if (noise_oracle_basis_mean_path_.empty() || noise_oracle_basis_path_.empty()) {
                throw Ort::Exception("Oracle coeff cached noise requires basis asset paths", ORT_INVALID_ARGUMENT);
            }
            if (noise_oracle_global_coeff_.size() != static_cast<size_t>(noise_oracle_basis_rank_)) {
                throw Ort::Exception("Oracle coeff cached noise global coeff size mismatch", ORT_INVALID_ARGUMENT);
            }
            const size_t expect_local = static_cast<size_t>(noise_oracle_residual_tiles_)
                * static_cast<size_t>(noise_oracle_residual_tiles_)
                * static_cast<size_t>(noise_oracle_residual_freq_h_)
                * static_cast<size_t>(noise_oracle_residual_freq_w_);
            if (noise_oracle_local_coeff_.size() != expect_local) {
                throw Ort::Exception("Oracle coeff cached noise local coeff size mismatch", ORT_INVALID_ARGUMENT);
            }
            oracle_basis_mean_host_ = LoadFloatBinaryFile(noise_oracle_basis_mean_path_, kTableSize);
            oracle_basis_host_ = LoadFloatBinaryFile(
                noise_oracle_basis_path_,
                static_cast<size_t>(noise_oracle_basis_rank_) * kTableSize);
        }

        CopyHostVectorToDevice(scale_a_arr_, d_scale_a_arr_);
        CopyHostVectorToDevice(scale_b_arr_, d_scale_b_arr_);
        CopyHostVectorToDevice(zp_a_arr_, d_zp_a_arr_);
        CopyHostVectorToDevice(zp_b_arr_, d_zp_b_arr_);
        CopyHostVectorToDevice(noise_p0_grid_, d_noise_p0_grid_);
        CopyHostVectorToDevice(noise_mix_grid_, d_noise_mix_grid_);
        CopyHostVectorToDevice(noise_atom1_grid_, d_noise_atom1_grid_);
        CopyHostVectorToDevice(noise_atom2_grid_, d_noise_atom2_grid_);
        CopyHostVectorToDevice(lut_delta_host_, d_lut_delta_);
    }

    ~ApproxMatMulCudaKernel() {
        if (d_scale_a_arr_) cudaFree(d_scale_a_arr_);
        if (d_scale_b_arr_) cudaFree(d_scale_b_arr_);
        if (d_zp_a_arr_) cudaFree(d_zp_a_arr_);
        if (d_zp_b_arr_) cudaFree(d_zp_b_arr_);
        if (d_noise_p0_grid_) cudaFree(d_noise_p0_grid_);
        if (d_noise_mix_grid_) cudaFree(d_noise_mix_grid_);
        if (d_noise_atom1_grid_) cudaFree(d_noise_atom1_grid_);
        if (d_noise_atom2_grid_) cudaFree(d_noise_atom2_grid_);
        if (d_lut_delta_) cudaFree(d_lut_delta_);
        if (d_oracle_basis_mean_) cudaFree(d_oracle_basis_mean_);
        if (d_oracle_basis_) cudaFree(d_oracle_basis_);
        if (d_oracle_global_coeff_) cudaFree(d_oracle_global_coeff_);
        if (d_oracle_local_coeff_) cudaFree(d_oracle_local_coeff_);
        if (d_oracle_error_grid_) cudaFree(d_oracle_error_grid_);
    }

    void Compute(OrtKernelContext* context) {
        const OrtValue* a_value = nullptr;
        const OrtValue* b_value = nullptr;
        const OrtValue* lut_value = nullptr;
        const OrtValue* base_value = nullptr;
        const OrtValue* a_code_value = nullptr;
        const OrtValue* b_code_value = nullptr;
        ThrowOnStatus(api_->KernelContext_GetInput(context, 0, &a_value));
        ThrowOnStatus(api_->KernelContext_GetInput(context, 1, &b_value));
        size_t input_count = 0;
        ThrowOnStatus(api_->KernelContext_GetInputCount(context, &input_count));
        if (input_count > 2) {
            ThrowOnStatus(api_->KernelContext_GetInput(context, 2, &lut_value));
        }
        if (base_input_) {
            if (input_count <= 3) {
                throw Ort::Exception("ApproxMatMulCuda base_input requires a fourth input", ORT_INVALID_ARGUMENT);
            }
            ThrowOnStatus(api_->KernelContext_GetInput(context, 3, &base_value));
            if (base_value == nullptr) {
                throw Ort::Exception("ApproxMatMulCuda base_input is missing", ORT_INVALID_ARGUMENT);
            }
        }
        if (input_count > 4) {
            ThrowOnStatus(api_->KernelContext_GetInput(context, 4, &a_code_value));
        }
        if (input_count > 5) {
            ThrowOnStatus(api_->KernelContext_GetInput(context, 5, &b_code_value));
        }
        if ((a_code_value == nullptr) != (b_code_value == nullptr)) {
            throw Ort::Exception("ApproxMatMulCuda code inputs must be supplied as a pair", ORT_INVALID_ARGUMENT);
        }
        if (signed_code_lut_ && a_code_value == nullptr) {
            throw Ort::Exception("ApproxMatMulCudaS8Lut requires paired INT8 QDQ code inputs", ORT_INVALID_ARGUMENT);
        }

        auto info_deleter = [this](OrtTensorTypeAndShapeInfo* ptr) {
            if (ptr) api_->ReleaseTensorTypeAndShapeInfo(ptr);
        };

        OrtTensorTypeAndShapeInfo* raw_a_info = nullptr;
        OrtTensorTypeAndShapeInfo* raw_b_info = nullptr;
        OrtTensorTypeAndShapeInfo* raw_lut_info = nullptr;
        OrtTensorTypeAndShapeInfo* raw_base_info = nullptr;
        ThrowOnStatus(api_->GetTensorTypeAndShape(a_value, &raw_a_info));
        ThrowOnStatus(api_->GetTensorTypeAndShape(b_value, &raw_b_info));
        std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(info_deleter)> a_info(raw_a_info, info_deleter);
        std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(info_deleter)> b_info(raw_b_info, info_deleter);
        std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(info_deleter)> lut_info(nullptr, info_deleter);
        std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(info_deleter)> base_info(nullptr, info_deleter);
        if (lut_value != nullptr) {
            ThrowOnStatus(api_->GetTensorTypeAndShape(lut_value, &raw_lut_info));
            lut_info.reset(raw_lut_info);
        }
        if (base_value != nullptr) {
            ThrowOnStatus(api_->GetTensorTypeAndShape(base_value, &raw_base_info));
            base_info.reset(raw_base_info);
        }

        size_t a_rank = 0;
        size_t b_rank = 0;
        ThrowOnStatus(api_->GetDimensionsCount(a_info.get(), &a_rank));
        ThrowOnStatus(api_->GetDimensionsCount(b_info.get(), &b_rank));
        if (a_rank == 0 || b_rank == 0) {
            throw Ort::Exception("ApproxMatMulCuda expects rank >= 1 inputs", ORT_INVALID_ARGUMENT);
        }

        std::vector<int64_t> a_shape(a_rank);
        std::vector<int64_t> b_shape(b_rank);
        ThrowOnStatus(api_->GetDimensions(a_info.get(), a_shape.data(), a_shape.size()));
        ThrowOnStatus(api_->GetDimensions(b_info.get(), b_shape.data(), b_shape.size()));

        const bool a_was_vector = (a_shape.size() == 1);
        const bool b_was_vector = (b_shape.size() == 1);
        std::vector<int64_t> a_promoted = PromoteInputShape(a_shape, true);
        std::vector<int64_t> b_promoted = PromoteInputShape(b_shape, false);
        if (a_promoted.size() > static_cast<size_t>(kMaxRank) || b_promoted.size() > static_cast<size_t>(kMaxRank)) {
            throw Ort::Exception("ApproxMatMulCuda only supports promoted rank <= 8", ORT_INVALID_ARGUMENT);
        }

        const int64_t m_dim = trans_a_ ? a_promoted.back() : a_promoted[a_promoted.size() - 2];
        const int64_t k_dim_a = trans_a_ ? a_promoted[a_promoted.size() - 2] : a_promoted.back();
        const int64_t k_dim_b = trans_b_ ? b_promoted.back() : b_promoted[b_promoted.size() - 2];
        const int64_t n_dim = trans_b_ ? b_promoted[b_promoted.size() - 2] : b_promoted.back();
        if (k_dim_a != k_dim_b) {
            throw Ort::Exception("ApproxMatMulCuda got incompatible K dimensions", ORT_INVALID_ARGUMENT);
        }
        if (m_dim < 0 || n_dim < 0 || k_dim_a < 0) {
            throw Ort::Exception("ApproxMatMulCuda does not support negative dimensions", ORT_INVALID_ARGUMENT);
        }

        const std::vector<int64_t> out_batch_dims = BroadcastBatchDims(
            std::vector<int64_t>(a_promoted.begin(), a_promoted.end() - 2),
            std::vector<int64_t>(b_promoted.begin(), b_promoted.end() - 2));
        if (out_batch_dims.size() > static_cast<size_t>(kMaxBatchRank)) {
            throw Ort::Exception("ApproxMatMulCuda only supports batch rank <= 6", ORT_INVALID_ARGUMENT);
        }

        std::vector<int64_t> output_shape = out_batch_dims;
        output_shape.push_back(m_dim);
        output_shape.push_back(n_dim);
        if (a_was_vector) {
            output_shape.erase(output_shape.end() - 2);
        }
        if (b_was_vector) {
            output_shape.pop_back();
        }
        if (base_info != nullptr) {
            size_t base_rank = 0;
            ThrowOnStatus(api_->GetDimensionsCount(base_info.get(), &base_rank));
            std::vector<int64_t> base_shape(base_rank);
            ThrowOnStatus(api_->GetDimensions(base_info.get(), base_shape.data(), base_shape.size()));
            if (base_shape != output_shape) {
                throw Ort::Exception("ApproxMatMulCuda exact baseline shape does not match MatMul output", ORT_INVALID_ARGUMENT);
            }
        }
        if (signed_code_lut_) {
            const auto validate_code_shape = [this, &info_deleter](
                                                 const OrtValue* code_value,
                                                 const std::vector<int64_t>& expected_shape,
                                                 const char* label) {
                OrtTensorTypeAndShapeInfo* raw_info = nullptr;
                ThrowOnStatus(api_->GetTensorTypeAndShape(code_value, &raw_info));
                std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(info_deleter)> code_info(raw_info, info_deleter);
                size_t rank = 0;
                ThrowOnStatus(api_->GetDimensionsCount(code_info.get(), &rank));
                std::vector<int64_t> shape(rank);
                ThrowOnStatus(api_->GetDimensions(code_info.get(), shape.data(), shape.size()));
                if (shape != expected_shape) {
                    throw Ort::Exception(
                        (std::string("ApproxMatMulCudaS8Lut ") + label + " code shape does not match DequantizeLinear output").c_str(),
                        ORT_INVALID_ARGUMENT);
                }
            };
            validate_code_shape(a_code_value, a_shape, "A");
            validate_code_shape(b_code_value, b_shape, "B");
        }

        const int32_t axis_a = NormalizeAxisForPromotedInput(raw_axis_a_, a_shape.size(), true);
        const int32_t axis_b = NormalizeAxisForPromotedInput(raw_axis_b_, b_shape.size(), false);
        ValidateAxisArrayCompatibility(a_promoted, axis_a, scale_a_arr_, zp_a_arr_, "A");
        ValidateAxisArrayCompatibility(b_promoted, axis_b, scale_b_arr_, zp_b_arr_, "B");

        const TensorLayout a_layout = BuildTensorLayout(a_promoted, out_batch_dims, axis_a, trans_a_);
        const TensorLayout b_layout = BuildTensorLayout(b_promoted, out_batch_dims, axis_b, trans_b_);
        const int64_t batch_total = Product(out_batch_dims);

        OrtValue* output_value = nullptr;
        ThrowOnStatus(api_->KernelContext_GetOutput(
            context,
            0,
            output_shape.empty() ? nullptr : output_shape.data(),
            output_shape.size(),
            &output_value));

        void* raw_a = nullptr;
        void* raw_b = nullptr;
        void* raw_base = nullptr;
        void* raw_y = nullptr;
        void* raw_lut = nullptr;
        void* raw_a_code = nullptr;
        void* raw_b_code = nullptr;
        ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(a_value), &raw_a));
        ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(b_value), &raw_b));
        ThrowOnStatus(api_->GetTensorMutableData(output_value, &raw_y));
        if (base_value != nullptr) {
            ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(base_value), &raw_base));
        }
        if (a_code_value != nullptr) {
            ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(a_code_value), &raw_a_code));
            ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(b_code_value), &raw_b_code));
        }
        if (lut_value != nullptr) {
            size_t lut_rank = 0;
            ThrowOnStatus(api_->GetDimensionsCount(lut_info.get(), &lut_rank));
            std::vector<int64_t> lut_shape(lut_rank);
            ThrowOnStatus(api_->GetDimensions(lut_info.get(), lut_shape.data(), lut_shape.size()));
            int64_t lut_elements = 1;
            for (int64_t dim : lut_shape) {
                if (dim < 0) {
                    throw Ort::Exception("ApproxMatMulCuda LUT input does not support negative dimensions", ORT_INVALID_ARGUMENT);
                }
                lut_elements *= dim;
            }
            if (lut_elements != 256 * 256) {
                throw Ort::Exception("ApproxMatMulCuda LUT input must contain exactly 65536 int32 values", ORT_INVALID_ARGUMENT);
            }
            ThrowOnStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(lut_value), &raw_lut));
        }

        void* raw_stream = nullptr;
        ThrowOnStatus(api_->KernelContext_GetGPUComputeStream(context, &raw_stream));
        if (raw_stream == nullptr) {
            throw Ort::Exception("Failed to fetch CUDA compute stream from ORT kernel context", ORT_RUNTIME_EXCEPTION);
        }
        cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
        EnsureOracleCacheReady(stream);
        const int32_t* lut_delta_ptr = (raw_lut != nullptr ? static_cast<const int32_t*>(raw_lut) : d_lut_delta_);
        const bool use_lut_table = (lut_delta_ptr != nullptr) && (use_lut_table_ || raw_lut != nullptr);
        if (signed_code_lut_ && !use_lut_table) {
            throw Ort::Exception("ApproxMatMulCudaS8Lut requires an INT32 signed LUT delta", ORT_INVALID_ARGUMENT);
        }

        const bool can_use_tiled_fast_path = CanUseTiledFastPath(a_shape, b_shape, a_layout, b_layout, axis_a, axis_b);
        const bool can_use_flattened_static_rhs_s8_fast_path =
            signed_code_lut_ &&
            can_use_tiled_fast_path &&
            CanUseFlattenedStaticRhsS8FastPath(a_shape, b_shape, out_batch_dims, a_layout, b_layout);
        const bool can_use_u8_residual_fast_path =
            !signed_code_lut_ && base_input_ && raw_a_code != nullptr && raw_b_code != nullptr && can_use_tiled_fast_path;
        if (signed_code_lut_ && can_use_tiled_fast_path) {
            constexpr int kTileM = kS8ResidualTileM;
            constexpr int kTileN = kS8ResidualTileN;
            constexpr int kTileK = kS8ResidualTileK;
            dim3 block(kTileN, kTileM);
            dim3 tiled_grid(
                static_cast<unsigned>((n_dim + kTileN - 1) / kTileN),
                static_cast<unsigned>((m_dim + kTileM - 1) / kTileM),
                static_cast<unsigned>(batch_total));
            const int64_t int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
            const bool can_use_int32_residual_accumulator =
                s8_residual_abs_bound_ >= 0 &&
                k_dim_a >= 0 &&
                (s8_residual_abs_bound_ == 0 ||
                 (s8_residual_abs_bound_ <= int32_max &&
                  k_dim_a <= int32_max / s8_residual_abs_bound_));
            if (can_use_flattened_static_rhs_s8_fast_path) {
                const int64_t flat_m_total = batch_total * m_dim;
                dim3 flattened_static_rhs_grid(
                    static_cast<unsigned>(flat_m_total / kTileM +
                                          static_cast<int64_t>(flat_m_total % kTileM != 0)),
                    static_cast<unsigned>(n_dim / kTileN +
                                          static_cast<int64_t>(n_dim % kTileN != 0)),
                    1);
                if (can_use_int32_residual_accumulator) {
                    approx_matmul_flattened_static_rhs_s8_residual_kernel<int32_t, kTileM, kTileN, kTileK><<<
                        flattened_static_rhs_grid,
                        block,
                        0,
                        stream>>>(
                        static_cast<const int8_t*>(raw_a_code),
                        static_cast<const int8_t*>(raw_b_code),
                        static_cast<const float*>(raw_base),
                        static_cast<float*>(raw_y),
                        flat_m_total,
                        k_dim_a,
                        n_dim,
                        scale_a_,
                        zp_a_,
                        scale_b_,
                        d_scale_b_arr_,
                        zp_b_,
                        d_zp_b_arr_,
                        alpha_,
                        lut_delta_ptr);
                } else {
                    approx_matmul_flattened_static_rhs_s8_residual_kernel<int64_t, kTileM, kTileN, kTileK><<<
                        flattened_static_rhs_grid,
                        block,
                        0,
                        stream>>>(
                        static_cast<const int8_t*>(raw_a_code),
                        static_cast<const int8_t*>(raw_b_code),
                        static_cast<const float*>(raw_base),
                        static_cast<float*>(raw_y),
                        flat_m_total,
                        k_dim_a,
                        n_dim,
                        scale_a_,
                        zp_a_,
                        scale_b_,
                        d_scale_b_arr_,
                        zp_b_,
                        d_zp_b_arr_,
                        alpha_,
                        lut_delta_ptr);
                }
            } else if (can_use_int32_residual_accumulator) {
                approx_matmul_tiled_s8_residual_kernel<int32_t, kTileM, kTileN, kTileK><<<tiled_grid, block, 0, stream>>>(
                    static_cast<const int8_t*>(raw_a_code),
                    static_cast<const int8_t*>(raw_b_code),
                    static_cast<const float*>(raw_base),
                    static_cast<float*>(raw_y),
                    batch_total,
                    m_dim,
                    k_dim_a,
                    n_dim,
                    a_layout,
                    b_layout,
                    scale_a_,
                    zp_a_,
                    scale_b_,
                    d_scale_b_arr_,
                    zp_b_,
                    d_zp_b_arr_,
                    alpha_,
                    lut_delta_ptr);
            } else {
                approx_matmul_tiled_s8_residual_kernel<int64_t, kTileM, kTileN, kTileK><<<tiled_grid, block, 0, stream>>>(
                    static_cast<const int8_t*>(raw_a_code),
                    static_cast<const int8_t*>(raw_b_code),
                    static_cast<const float*>(raw_base),
                    static_cast<float*>(raw_y),
                    batch_total,
                    m_dim,
                    k_dim_a,
                    n_dim,
                    a_layout,
                    b_layout,
                    scale_a_,
                    zp_a_,
                    scale_b_,
                    d_scale_b_arr_,
                    zp_b_,
                    d_zp_b_arr_,
                    alpha_,
                    lut_delta_ptr);
            }
        } else if (signed_code_lut_) {
            constexpr int kThreads = 256;
            const int64_t total = batch_total * m_dim * n_dim;
            const int64_t blocks = (total + kThreads - 1) / kThreads;
            approx_matmul_s8_residual_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
                static_cast<const int8_t*>(raw_a_code),
                static_cast<const int8_t*>(raw_b_code),
                static_cast<const float*>(raw_base),
                static_cast<float*>(raw_y),
                batch_total,
                m_dim,
                k_dim_a,
                n_dim,
                a_layout,
                b_layout,
                scale_a_,
                d_scale_a_arr_,
                zp_a_,
                d_zp_a_arr_,
                scale_b_,
                d_scale_b_arr_,
                zp_b_,
                d_zp_b_arr_,
                alpha_,
                lut_delta_ptr);
        } else if (can_use_u8_residual_fast_path) {
            constexpr int kTileM = 16;
            constexpr int kTileN = 16;
            constexpr int kTileK = 32;
            dim3 block(kTileN, kTileM);
            dim3 grid(
                static_cast<unsigned>((n_dim + kTileN - 1) / kTileN),
                static_cast<unsigned>((m_dim + kTileM - 1) / kTileM),
                static_cast<unsigned>(batch_total));
            approx_matmul_tiled_u8_residual_kernel<kTileM, kTileN, kTileK><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(raw_a_code),
                static_cast<const uint8_t*>(raw_b_code),
                static_cast<const float*>(raw_base),
                static_cast<float*>(raw_y),
                batch_total,
                m_dim,
                k_dim_a,
                n_dim,
                a_layout,
                b_layout,
                qmin_a_,
                qmin_b_,
                scale_a_,
                scale_b_,
                d_scale_b_arr_,
                alpha_,
                kind_,
                lut_delta_ptr,
                use_lut_table ? 1 : 0);
        } else if (can_use_tiled_fast_path) {
            constexpr int kTileM = 16;
            constexpr int kTileN = 16;
            constexpr int kTileK = 32;
            dim3 block(kTileN, kTileM);
            dim3 grid(
                static_cast<unsigned>((n_dim + kTileN - 1) / kTileN),
                static_cast<unsigned>((m_dim + kTileM - 1) / kTileM),
                static_cast<unsigned>(batch_total));
            approx_matmul_tiled_fast_kernel<kTileM, kTileN, kTileK><<<grid, block, 0, stream>>>(
                static_cast<const float*>(raw_a),
                static_cast<const float*>(raw_b),
                static_cast<const float*>(raw_base),
                static_cast<float*>(raw_y),
                batch_total,
                m_dim,
                k_dim_a,
                n_dim,
                a_layout,
                b_layout,
                scale_a_,
                zp_a_,
                qmin_a_,
                qmax_a_,
                scale_b_,
                d_scale_b_arr_,
                zp_b_,
                d_zp_b_arr_,
                qmin_b_,
                qmax_b_,
                alpha_,
                kind_,
                lut_delta_ptr,
                use_lut_table ? 1 : 0,
                base_input_ ? 1 : 0);
        } else {
            const auto total = batch_total * m_dim * n_dim;
            dim3 block(256);
            dim3 grid(static_cast<unsigned>((total + block.x - 1) / block.x));
            approx_matmul_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(raw_a),
                static_cast<const float*>(raw_b),
                static_cast<const float*>(raw_base),
                static_cast<float*>(raw_y),
                batch_total,
                m_dim,
                k_dim_a,
                n_dim,
                a_layout,
                b_layout,
                scale_a_,
                d_scale_a_arr_,
                zp_a_,
                d_zp_a_arr_,
                qmin_a_,
                qmax_a_,
                scale_b_,
                d_scale_b_arr_,
                zp_b_,
                d_zp_b_arr_,
                qmin_b_,
                qmax_b_,
                alpha_,
                kind_,
                d_noise_p0_grid_,
                d_noise_mix_grid_,
                d_noise_atom1_grid_,
                d_noise_atom2_grid_,
                d_oracle_error_grid_,
                lut_delta_ptr,
                noise_seed_,
                noise_strength_,
                noise_force_nonzero_ ? 1 : 0,
                (noise_enable_ && noise_direct_condmix_) ? 1 : 0,
                (noise_enable_ && noise_oracle_surface_coeff_cached_) ? 1 : 0,
                use_lut_table ? 1 : 0,
                base_input_ ? 1 : 0);
        }

        // CUDA EP schedules dependent nodes on this stream.  A per-MatMul
        // host-side synchronization serializes all 72 injected ViT nodes;
        // ORT synchronizes when the caller fetches the completed outputs.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw Ort::Exception("ApproxMatMulCuda kernel failed", ORT_RUNTIME_EXCEPTION);
        }
    }

private:
    const OrtApi* api_{};

    float scale_a_{1.0f};
    float scale_b_{1.0f};
    float alpha_{1.0f};
    int32_t zp_a_{0};
    int32_t zp_b_{0};
    int32_t qmin_a_{0};
    int32_t qmax_a_{255};
    int32_t qmin_b_{0};
    int32_t qmax_b_{255};
    int64_t raw_axis_a_{std::numeric_limits<int64_t>::min()};
    int64_t raw_axis_b_{std::numeric_limits<int64_t>::min()};
    bool trans_a_{false};
    bool trans_b_{false};
    bool base_input_{false};
    MultiplierKind kind_{MultiplierKind::kAkbar1};
    bool noise_enable_{false};
    uint64_t noise_seed_{7};
    float noise_strength_{1.0f};
    bool noise_force_nonzero_{true};
    bool noise_direct_condmix_{false};
    bool noise_oracle_surface_coeff_cached_{false};
    bool use_lut_table_{false};
    bool signed_code_lut_{false};
    int64_t s8_residual_abs_bound_{-1};
    std::string noise_oracle_basis_mean_path_;
    std::string noise_oracle_basis_path_;
    std::string lut_delta_path_;
    int32_t noise_oracle_basis_side_{256};
    int32_t noise_oracle_basis_rank_{0};
    int32_t noise_oracle_residual_tiles_{0};
    int32_t noise_oracle_residual_freq_h_{0};
    int32_t noise_oracle_residual_freq_w_{0};
    std::vector<float> noise_oracle_global_coeff_;
    std::vector<float> noise_oracle_local_coeff_;

    std::vector<float> scale_a_arr_;
    std::vector<float> scale_b_arr_;
    std::vector<int32_t> zp_a_arr_;
    std::vector<int32_t> zp_b_arr_;
    std::vector<float> noise_p0_grid_;
    std::vector<float> noise_mix_grid_;
    std::vector<float> noise_atom1_grid_;
    std::vector<float> noise_atom2_grid_;
    std::vector<int32_t> lut_delta_host_;
    std::vector<float> oracle_basis_mean_host_;
    std::vector<float> oracle_basis_host_;

    float* d_scale_a_arr_{nullptr};
    float* d_scale_b_arr_{nullptr};
    int32_t* d_zp_a_arr_{nullptr};
    int32_t* d_zp_b_arr_{nullptr};
    float* d_noise_p0_grid_{nullptr};
    float* d_noise_mix_grid_{nullptr};
    float* d_noise_atom1_grid_{nullptr};
    float* d_noise_atom2_grid_{nullptr};
    int32_t* d_lut_delta_{nullptr};
    float* d_oracle_basis_mean_{nullptr};
    float* d_oracle_basis_{nullptr};
    float* d_oracle_global_coeff_{nullptr};
    float* d_oracle_local_coeff_{nullptr};
    int32_t* d_oracle_error_grid_{nullptr};
    bool oracle_cache_ready_{false};

    float GetFloatAttributeOr(const OrtKernelInfo* info, const char* name, float default_value) const {
        float value = default_value;
        OrtStatus* status = api_->KernelInfoGetAttribute_float(info, name, &value);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        return value;
    }

    int64_t GetInt64AttributeOr(const OrtKernelInfo* info, const char* name, int64_t default_value) const {
        int64_t value = default_value;
        OrtStatus* status = api_->KernelInfoGetAttribute_int64(info, name, &value);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        return value;
    }

    std::string GetStringAttributeOr(const OrtKernelInfo* info, const char* name, const std::string& default_value) const {
        size_t length = 0;
        OrtStatus* status = api_->KernelInfoGetAttribute_string(info, name, nullptr, &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        std::string result(length, '\0');
        status = api_->KernelInfoGetAttribute_string(info, name, result.data(), &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        if (!result.empty() && result.back() == '\0') {
            result.pop_back();
        }
        return result;
    }

    std::vector<int64_t> GetIntsAttributeOr(
        const OrtKernelInfo* info,
        const char* name,
        const std::vector<int64_t>& default_value) const {
        size_t length = 0;
        OrtStatus* status = api_->KernelInfoGetAttributeArray_int64(info, name, nullptr, &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        if (length == 0) {
            return default_value;
        }
        std::vector<int64_t> values(length);
        status = api_->KernelInfoGetAttributeArray_int64(info, name, values.data(), &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        return values;
    }

    std::vector<float> GetFloatsAttributeOr(
        const OrtKernelInfo* info,
        const char* name,
        const std::vector<float>& default_value) const {
        size_t length = 0;
        OrtStatus* status = api_->KernelInfoGetAttributeArray_float(info, name, nullptr, &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        if (length == 0) {
            return default_value;
        }
        std::vector<float> values(length);
        status = api_->KernelInfoGetAttributeArray_float(info, name, values.data(), &length);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            return default_value;
        }
        return values;
    }

    float GetFloatAttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names,
        float default_value) const {
        const float sentinel = std::numeric_limits<float>::quiet_NaN();
        for (const char* name : names) {
            const float value = GetFloatAttributeOr(info, name, sentinel);
            if (!std::isnan(value)) {
                return value;
            }
        }
        return default_value;
    }

    int64_t GetInt64AttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names,
        int64_t default_value) const {
        const int64_t sentinel = std::numeric_limits<int64_t>::min();
        for (const char* name : names) {
            const int64_t value = GetInt64AttributeOr(info, name, sentinel);
            if (value != sentinel) {
                return value;
            }
        }
        return default_value;
    }

    int64_t GetOptionalInt64AttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names) const {
        return GetInt64AttributeAlias(info, names, std::numeric_limits<int64_t>::min());
    }

    bool GetBoolAttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names,
        bool default_value) const {
        return GetInt64AttributeAlias(info, names, default_value ? 1 : 0) != 0;
    }

    std::vector<int64_t> GetIntsAttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names,
        const std::vector<int64_t>& default_value) const {
        for (const char* name : names) {
            const std::vector<int64_t> values = GetIntsAttributeOr(info, name, {});
            if (!values.empty()) {
                return values;
            }
        }
        return default_value;
    }

    std::vector<float> GetFloatsAttributeAlias(
        const OrtKernelInfo* info,
        std::initializer_list<const char*> names,
        const std::vector<float>& default_value) const {
        for (const char* name : names) {
            const std::vector<float> values = GetFloatsAttributeOr(info, name, {});
            if (!values.empty()) {
                return values;
            }
        }
        return default_value;
    }

    template <typename T>
    static void CopyHostVectorToDevice(const std::vector<T>& host, T*& device_ptr) {
        if (host.empty()) {
            device_ptr = nullptr;
            return;
        }
        cudaMalloc(&device_ptr, sizeof(T) * host.size());
        cudaMemcpy(device_ptr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    }

    void ThrowOnStatus(OrtStatus* status) const {
        if (status != nullptr) {
            const char* msg = api_->GetErrorMessage(status);
            std::string message = msg != nullptr ? msg : "Ort API call failed";
            api_->ReleaseStatus(status);
            throw Ort::Exception(message.c_str(), ORT_RUNTIME_EXCEPTION);
        }
    }

    void EnsureOracleCacheReady(cudaStream_t stream) {
        if (!(noise_enable_ && noise_oracle_surface_coeff_cached_)) return;
        if (oracle_cache_ready_) return;

        constexpr size_t kTableSize = 256u * 256u;
        auto ensure_device_copy = [](float*& dst, const std::vector<float>& src) {
            if (dst != nullptr || src.empty()) return;
            cudaMalloc(&dst, sizeof(float) * src.size());
            cudaMemcpy(dst, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice);
        };

        ensure_device_copy(d_oracle_basis_mean_, oracle_basis_mean_host_);
        ensure_device_copy(d_oracle_basis_, oracle_basis_host_);
        ensure_device_copy(d_oracle_global_coeff_, noise_oracle_global_coeff_);
        ensure_device_copy(d_oracle_local_coeff_, noise_oracle_local_coeff_);
        if (d_oracle_error_grid_ == nullptr) {
            cudaMalloc(&d_oracle_error_grid_, sizeof(int32_t) * kTableSize);
        }

        dim3 block(256);
        dim3 grid(static_cast<unsigned>((kTableSize + block.x - 1) / block.x));
        decode_oracle_surface_error_grid_kernel<<<grid, block, 0, stream>>>(
            d_oracle_basis_mean_,
            d_oracle_basis_,
            d_oracle_global_coeff_,
            noise_oracle_basis_rank_,
            d_oracle_local_coeff_,
            noise_oracle_residual_tiles_,
            noise_oracle_residual_freq_h_,
            noise_oracle_residual_freq_w_,
            noise_strength_,
            noise_force_nonzero_ ? 1 : 0,
            d_oracle_error_grid_);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(stream);
        }
        if (err != cudaSuccess) {
            throw Ort::Exception("Failed to decode oracle surface coeff cache on CUDA", ORT_RUNTIME_EXCEPTION);
        }
        std::vector<float>().swap(oracle_basis_mean_host_);
        std::vector<float>().swap(oracle_basis_host_);
        std::vector<float>().swap(noise_oracle_global_coeff_);
        std::vector<float>().swap(noise_oracle_local_coeff_);
        oracle_cache_ready_ = true;
    }

    static std::vector<int64_t> PromoteInputShape(const std::vector<int64_t>& shape, bool lhs) {
        std::vector<int64_t> promoted = shape;
        if (shape.size() == 1) {
            if (lhs) promoted.insert(promoted.begin(), 1);
            else promoted.push_back(1);
        }
        return promoted;
    }

    static int32_t NormalizeAxisForPromotedInput(int64_t raw_axis, size_t original_rank, bool lhs) {
        if (raw_axis == std::numeric_limits<int64_t>::min()) {
            return -1;
        }
        int64_t axis = raw_axis;
        if (axis < 0) {
            axis += static_cast<int64_t>(original_rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(original_rank)) {
            throw Ort::Exception("ApproxMatMulCuda axis attribute is out of range", ORT_INVALID_ARGUMENT);
        }
        if (original_rank == 1 && lhs) {
            axis += 1;
        }
        return static_cast<int32_t>(axis);
    }

    static std::vector<int64_t> BroadcastBatchDims(
        const std::vector<int64_t>& a_batch,
        const std::vector<int64_t>& b_batch) {
        const size_t out_rank = std::max(a_batch.size(), b_batch.size());
        std::vector<int64_t> out(out_rank, 1);
        for (size_t i = 0; i < out_rank; ++i) {
            const int64_t a_dim = (i < out_rank - a_batch.size()) ? 1 : a_batch[i - (out_rank - a_batch.size())];
            const int64_t b_dim = (i < out_rank - b_batch.size()) ? 1 : b_batch[i - (out_rank - b_batch.size())];
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw Ort::Exception("ApproxMatMulCuda got non-broadcastable batch dimensions", ORT_INVALID_ARGUMENT);
            }
            out[i] = std::max(a_dim, b_dim);
        }
        return out;
    }

    static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        if (shape.empty()) {
            return strides;
        }
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
        }
        return strides;
    }

    static int64_t Product(const std::vector<int64_t>& dims) {
        int64_t value = 1;
        for (int64_t dim : dims) {
            value *= dim;
        }
        return value;
    }

    static void ValidateAxisArrayCompatibility(
        const std::vector<int64_t>& promoted_shape,
        int32_t axis,
        const std::vector<float>& scale_arr,
        const std::vector<int32_t>& zp_arr,
        const char* label) {
        const bool has_scale = !scale_arr.empty();
        const bool has_zp = !zp_arr.empty();
        if (!has_scale && !has_zp) {
            return;
        }
        if (axis < 0) {
            throw Ort::Exception(
                (std::string("ApproxMatMulCuda requires axis for per-axis quantization on input ") + label).c_str(),
                ORT_INVALID_ARGUMENT);
        }
        const int64_t axis_dim = promoted_shape[static_cast<size_t>(axis)];
        if (has_scale && static_cast<int64_t>(scale_arr.size()) != axis_dim) {
            throw Ort::Exception(
                (std::string("ApproxMatMulCuda scale array length mismatch on input ") + label).c_str(),
                ORT_INVALID_ARGUMENT);
        }
        if (has_zp && static_cast<int64_t>(zp_arr.size()) != axis_dim) {
            throw Ort::Exception(
                (std::string("ApproxMatMulCuda zero-point array length mismatch on input ") + label).c_str(),
                ORT_INVALID_ARGUMENT);
        }
    }

    bool CanUseTiledFastPath(
        const std::vector<int64_t>& a_shape,
        const std::vector<int64_t>& b_shape,
        const TensorLayout& a_layout,
        const TensorLayout& b_layout,
        int32_t axis_a,
        int32_t axis_b) const {
        if (noise_enable_) return false;
        if (trans_a_ || trans_b_) return false;
        if (a_shape.size() < 2 || b_shape.size() < 2) return false;
        if (axis_a >= 0) return false;
        if (!scale_a_arr_.empty() || !zp_a_arr_.empty()) return false;
        if (axis_b >= 0 && axis_b != b_layout.mat_axis1) return false;
        if (b_shape[static_cast<size_t>(b_layout.mat_axis1)] <= 0) return false;
        if (scale_b_arr_.empty() != zp_b_arr_.empty() && !zp_b_arr_.empty()) {
            return false;
        }
        return true;
    }

    bool CanUseFlattenedStaticRhsS8FastPath(
        const std::vector<int64_t>& a_shape,
        const std::vector<int64_t>& b_shape,
        const std::vector<int64_t>& out_batch_dims,
        const TensorLayout& a_layout,
        const TensorLayout& b_layout) const {
        // The raw ORT tensor buffers are contiguous.  Restrict this path to
        // the exact static-RHS layout where flattening [B, M] is equivalent
        // to the existing batch-aware offset calculation.
        if (a_shape.size() != 3 || b_shape.size() != 2 || out_batch_dims.size() != 1) return false;
        const int64_t batch_dim = a_shape[0];
        const int64_t m_dim = a_shape[1];
        const int64_t k_dim = a_shape[2];
        const int64_t n_dim = b_shape[1];
        if (batch_dim <= 0 || m_dim <= 0 || k_dim <= 0 || n_dim <= 0) return false;
        if (b_shape[0] != k_dim || out_batch_dims[0] != batch_dim) return false;
        const int64_t int64_max = std::numeric_limits<int64_t>::max();
        if (m_dim > int64_max / k_dim || batch_dim > int64_max / m_dim) return false;
        const int64_t matrix_elements = m_dim * k_dim;
        const int64_t flat_m_total = batch_dim * m_dim;
        const int64_t grid_x = flat_m_total / kS8ResidualTileM +
            static_cast<int64_t>(flat_m_total % kS8ResidualTileM != 0);
        const int64_t grid_y = n_dim / kS8ResidualTileN +
            static_cast<int64_t>(n_dim % kS8ResidualTileN != 0);
        if (grid_x > kCudaMaxGridX || grid_y > kCudaMaxGridY) return false;
        if (a_layout.rank != 3 || b_layout.rank != 2 ||
            a_layout.batch_rank != 1 || b_layout.batch_rank != 1 ||
            a_layout.mat_axis0 != 1 || a_layout.mat_axis1 != 2 ||
            b_layout.mat_axis0 != 0 || b_layout.mat_axis1 != 1) {
            return false;
        }
        if (a_layout.batch_dims[0] != batch_dim || b_layout.batch_dims[0] != batch_dim ||
            a_layout.batch_strides[0] != (batch_dim == 1 ? 0 : matrix_elements) ||
            b_layout.batch_strides[0] != 0) {
            return false;
        }
        return a_layout.strides[0] == matrix_elements &&
            a_layout.strides[1] == k_dim &&
            a_layout.strides[2] == 1 &&
            b_layout.strides[0] == n_dim &&
            b_layout.strides[1] == 1;
    }

    static TensorLayout BuildTensorLayout(
        const std::vector<int64_t>& promoted_shape,
        const std::vector<int64_t>& out_batch_dims,
        int32_t axis,
        bool trans) {
        TensorLayout layout{};
        layout.rank = static_cast<int32_t>(promoted_shape.size());
        layout.batch_rank = static_cast<int32_t>(out_batch_dims.size());
        layout.axis = axis;
        layout.mat_axis0 = trans ? (layout.rank - 1) : (layout.rank - 2);
        layout.mat_axis1 = trans ? (layout.rank - 2) : (layout.rank - 1);

        const std::vector<int64_t> strides = ComputeStrides(promoted_shape);
        for (int32_t i = 0; i < layout.rank; ++i) {
            layout.shape[i] = promoted_shape[static_cast<size_t>(i)];
            layout.strides[i] = strides[static_cast<size_t>(i)];
        }
        for (int32_t i = layout.rank; i < kMaxRank; ++i) {
            layout.shape[i] = 1;
            layout.strides[i] = 0;
        }

        const int32_t input_batch_rank = layout.rank - 2;
        const int32_t batch_offset = layout.batch_rank - input_batch_rank;
        for (int32_t i = 0; i < layout.batch_rank; ++i) {
            layout.batch_dims[i] = out_batch_dims[static_cast<size_t>(i)];
            if (i < batch_offset) {
                layout.batch_strides[i] = 0;
            } else {
                const int32_t input_axis = i - batch_offset;
                layout.batch_strides[i] = (promoted_shape[static_cast<size_t>(input_axis)] == 1)
                    ? 0
                    : strides[static_cast<size_t>(input_axis)];
            }
        }
        for (int32_t i = layout.batch_rank; i < kMaxBatchRank; ++i) {
            layout.batch_dims[i] = 1;
            layout.batch_strides[i] = 0;
        }
        return layout;
    }
};

class ApproxMatMulCudaOp : public Ort::CustomOpBase<ApproxMatMulCudaOp, ApproxMatMulCudaKernel> {
public:
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ApproxMatMulCudaKernel(api, info);
    }

    const char* GetName() const { return "ApproxMatMulCuda"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 6; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        if (index == 4 || index == 5) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return (index >= 2)
            ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL
            : OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
};

class ApproxMatMulCudaS8LutOp : public Ort::CustomOpBase<ApproxMatMulCudaS8LutOp, ApproxMatMulCudaKernel> {
public:
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ApproxMatMulCudaKernel(api, info, true);
    }

    const char* GetName() const { return "ApproxMatMulCudaS8Lut"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 6; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        if (index == 4 || index == 5) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return (index >= 2)
            ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL
            : OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
};

static ApproxMatMulCudaOp g_approx_matmul_cuda_op;
static ApproxMatMulCudaS8LutOp g_approx_matmul_cuda_s8_lut_op;

extern "C" const OrtCustomOp* GetApproxMatMulCudaCustomOp() {
    return static_cast<const OrtCustomOp*>(&g_approx_matmul_cuda_op);
}

extern "C" const OrtCustomOp* GetApproxMatMulCudaS8LutCustomOp() {
    return static_cast<const OrtCustomOp*>(&g_approx_matmul_cuda_s8_lut_op);
}

}  // namespace approx_matmul_cuda

extern "C" EA2_ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    OrtCustomOpDomain* domain = nullptr;
    OrtStatus* status = api->CreateCustomOpDomain("approxdomain", &domain);
    if (status != nullptr) {
        return status;
    }
    status = api->CustomOpDomain_Add(domain, approx_matmul_cuda::GetApproxMatMulCudaCustomOp());
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }
    status = api->CustomOpDomain_Add(domain, approx_matmul_cuda::GetApproxMatMulCudaS8LutCustomOp());
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }
    status = api->AddCustomOpDomain(options, domain);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }
    return nullptr;
}
