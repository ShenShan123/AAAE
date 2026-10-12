#include <cuda_runtime.h>
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#define CNN_ORT_EXPORT __declspec(dllexport)
#else
#define CNN_ORT_EXPORT __attribute__((visibility("default")))
#endif

namespace cnn_s8s8 {

constexpr const char* kDomain = "s8s8_sign_magnitude";
constexpr const char* kIndexEncoding = "signed_code_plus_128";

template <typename Accumulator>
__device__ __forceinline__ Accumulator ResidualForCodes(
    int32_t a,
    int32_t b,
    const int32_t* __restrict__ residual_lut,
    int32_t activation_zero_point,
    int32_t weight_zero_point) {
  if (a == activation_zero_point || b == weight_zero_point) return Accumulator{0};
  const int32_t lut_index = (a + 128) * 256 + (b + 128);
  return static_cast<Accumulator>(__ldg(residual_lut + lut_index));
}

template <typename Accumulator, int FixedKernelH, int FixedKernelW>
__global__ void AddSignedResidualConvSpatial(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ exact,
    const int32_t* __restrict__ residual_lut,
    float* __restrict__ output,
    int channels,
    int height,
    int width,
    int out_channels,
    int channels_per_group,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int pad_top,
    int pad_left,
    int groups,
    float activation_scale,
    int32_t activation_zero_point,
    const float* __restrict__ weight_scales,
    float scalar_weight_scale,
    const int32_t* __restrict__ weight_zero_points,
    int32_t scalar_weight_zero_point) {
  const int spatial = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int spatial_count = output_h * output_w;
  if (spatial >= spatial_count) return;
  const int n = static_cast<int>(blockIdx.z);
  const int m = static_cast<int>(blockIdx.y);
  const int oh = spatial / output_w;
  const int ow = spatial - oh * output_w;
  const int outputs_per_group = out_channels / groups;
  const int channel_base = (m / outputs_per_group) * channels_per_group;
  const int32_t weight_zero_point =
      weight_zero_points == nullptr ? scalar_weight_zero_point : weight_zero_points[m];
  const int resolved_kernel_h = FixedKernelH > 0 ? FixedKernelH : kernel_h;
  const int resolved_kernel_w = FixedKernelW > 0 ? FixedKernelW : kernel_w;
  const int input_channel_stride = height * width;
  const int weight_channel_stride = resolved_kernel_h * resolved_kernel_w;
  const int input_channel_base =
      (n * channels + channel_base) * input_channel_stride;
  const int weight_output_base = m * channels_per_group * weight_channel_stride;
  Accumulator residual_sum = 0;
#pragma unroll
  for (int kh = 0; kh < resolved_kernel_h; ++kh) {
    const int ih = oh * stride_h - pad_top + kh * dilation_h;
    if (ih < 0 || ih >= height) continue;
    const int input_row_base = input_channel_base + ih * width;
    const int weight_row_base = weight_output_base + kh * resolved_kernel_w;
#pragma unroll
    for (int kw = 0; kw < resolved_kernel_w; ++kw) {
      const int iw = ow * stride_w - pad_left + kw * dilation_w;
      if (iw < 0 || iw >= width) continue;
      int x_index = input_row_base + iw;
      int w_index = weight_row_base + kw;
      for (int local_c = 0; local_c < channels_per_group; ++local_c) {
        const int32_t a = static_cast<int32_t>(x[x_index]);
        const int32_t b = static_cast<int32_t>(w[w_index]);
        residual_sum += ResidualForCodes<Accumulator>(
            a, b, residual_lut, activation_zero_point, weight_zero_point);
        x_index += input_channel_stride;
        w_index += weight_channel_stride;
      }
    }
  }
  const int output_index = ((n * out_channels + m) * output_h + oh) * output_w + ow;
  const float weight_scale =
      weight_scales == nullptr ? scalar_weight_scale : weight_scales[m];
  output[output_index] = exact[output_index] +
                         static_cast<float>(residual_sum) * activation_scale * weight_scale;
}

class SignedResidualConvKernel {
 public:
  SignedResidualConvKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(&api) {
    scale_x_ = GetFloat(info, "scale_x");
    scale_w_ = GetFloat(info, "scale_w");
    zp_x_ = GetInt(info, "zp_x");
    zp_w_ = GetInt(info, "zp_w");
    semantic_zero_mask_ = GetInt(info, "semantic_zero_mask");
    lut_domain_ = GetString(info, "lut_domain");
    index_encoding_ = GetString(info, "lut_index_encoding");
    strides_ = GetInts(info, "strides", {1, 1});
    dilations_ = GetInts(info, "dilations", {1, 1});
    pads_ = GetInts(info, "pads", {0, 0, 0, 0});
    group_ = GetIntOr(info, "group", 1);
    residual_abs_bound_ = GetIntOr(info, "residual_abs_bound", -1);
    scale_w_arr_ = GetFloats(info, "scale_w_arr", {});
    const auto raw_zp_w_arr = GetInts(info, "zp_w_arr", {});

    if (lut_domain_ != kDomain || index_encoding_ != kIndexEncoding || semantic_zero_mask_ != 1) {
      throw Ort::Exception("ApproxConvCudaS8Lut received an incompatible signed LUT contract", ORT_INVALID_ARGUMENT);
    }
    if (zp_x_ < -128 || zp_x_ > 127 || zp_w_ < -128 || zp_w_ > 127 ||
        std::any_of(raw_zp_w_arr.begin(), raw_zp_w_arr.end(),
                    [](int64_t value) { return value < -128 || value > 127; })) {
      throw Ort::Exception("ApproxConvCudaS8Lut zero points must be valid INT8 codes", ORT_INVALID_ARGUMENT);
    }
    weight_zero_points_.reserve(raw_zp_w_arr.size());
    for (const int64_t value : raw_zp_w_arr) {
      weight_zero_points_.push_back(static_cast<int32_t>(value));
    }
    if (!(std::isfinite(scale_x_) && scale_x_ > 0.0f &&
          std::isfinite(scale_w_) && scale_w_ > 0.0f) ||
        std::any_of(scale_w_arr_.begin(), scale_w_arr_.end(),
                    [](float value) { return !(std::isfinite(value) && value > 0.0f); })) {
      throw Ort::Exception("ApproxConvCudaS8Lut requires finite positive quantization scales", ORT_INVALID_ARGUMENT);
    }
    if (strides_.size() != 2 || dilations_.size() != 2 || pads_.size() != 4 ||
        strides_[0] <= 0 || strides_[1] <= 0 || dilations_[0] <= 0 || dilations_[1] <= 0 ||
        std::any_of(pads_.begin(), pads_.end(), [](int64_t value) { return value < 0; }) || group_ <= 0) {
      throw Ort::Exception("ApproxConvCudaS8Lut received invalid two-dimensional Conv attributes", ORT_INVALID_ARGUMENT);
    }
    if (residual_abs_bound_ < -1 || residual_abs_bound_ > std::numeric_limits<int32_t>::max()) {
      throw Ort::Exception("ApproxConvCudaS8Lut received an invalid residual_abs_bound", ORT_INVALID_ARGUMENT);
    }
  }

  ~SignedResidualConvKernel() {
    if (device_weight_scales_ != nullptr) {
      const int previous_device = CurrentDeviceNoThrow();
      if (scale_device_ >= 0) cudaSetDevice(scale_device_);
      cudaFree(device_weight_scales_);
      if (device_weight_zero_points_ != nullptr) cudaFree(device_weight_zero_points_);
      if (previous_device >= 0 && previous_device != scale_device_) cudaSetDevice(previous_device);
    } else if (device_weight_zero_points_ != nullptr) {
      const int previous_device = CurrentDeviceNoThrow();
      if (scale_device_ >= 0) cudaSetDevice(scale_device_);
      cudaFree(device_weight_zero_points_);
      if (previous_device >= 0 && previous_device != scale_device_) cudaSetDevice(previous_device);
    }
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* x_value = Input(context, 0);
    const OrtValue* w_value = Input(context, 1);
    const OrtValue* exact_value = Input(context, 2);
    const OrtValue* lut_value = Input(context, 3);
    const std::vector<int64_t> x_shape = Shape(x_value);
    const std::vector<int64_t> w_shape = Shape(w_value);
    const std::vector<int64_t> exact_shape = Shape(exact_value);
    const std::vector<int64_t> lut_shape = Shape(lut_value);
    if (x_shape.size() != 4 || w_shape.size() != 4 || exact_shape.size() != 4 ||
        lut_shape != std::vector<int64_t>({256, 256})) {
      throw Ort::Exception(
          "ApproxConvCudaS8Lut expects X/W/exact rank 4 and residual LUT [256,256]",
          ORT_INVALID_ARGUMENT);
    }
    if (std::any_of(x_shape.begin(), x_shape.end(), [](int64_t value) { return value <= 0; }) ||
        std::any_of(w_shape.begin(), w_shape.end(), [](int64_t value) { return value <= 0; }) ||
        std::any_of(exact_shape.begin(), exact_shape.end(), [](int64_t value) { return value <= 0; })) {
      throw Ort::Exception("ApproxConvCudaS8Lut received a nonpositive runtime dimension", ORT_INVALID_ARGUMENT);
    }

    const int64_t n = x_shape[0];
    const int64_t channels = x_shape[1];
    const int64_t height = x_shape[2];
    const int64_t width = x_shape[3];
    const int64_t out_channels = w_shape[0];
    const int64_t channels_per_group = w_shape[1];
    const int64_t kernel_h = w_shape[2];
    const int64_t kernel_w = w_shape[3];
    const int64_t grouped_channels = CheckedProduct({channels_per_group, group_});
    if (grouped_channels != channels || out_channels % group_ != 0 ||
        exact_shape[0] != n || exact_shape[1] != out_channels) {
      throw Ort::Exception("ApproxConvCudaS8Lut received incompatible grouped-Conv shapes", ORT_INVALID_ARGUMENT);
    }
    if (!scale_w_arr_.empty() && scale_w_arr_.size() != static_cast<size_t>(out_channels)) {
      throw Ort::Exception("ApproxConvCudaS8Lut weight-scale count does not match output channels", ORT_INVALID_ARGUMENT);
    }
    if (!weight_zero_points_.empty() &&
        weight_zero_points_.size() != static_cast<size_t>(out_channels)) {
      throw Ort::Exception("ApproxConvCudaS8Lut weight zero-point count does not match output channels", ORT_INVALID_ARGUMENT);
    }
    const int64_t residual_terms = CheckedProduct({channels_per_group, kernel_h, kernel_w});
    if (residual_terms > std::numeric_limits<int64_t>::max() /
                             static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      throw Ort::Exception("ApproxConvCudaS8Lut residual accumulation could overflow int64", ORT_INVALID_ARGUMENT);
    }
    const int64_t x_elements = CheckedProduct({n, channels, height, width});
    const int64_t w_elements = CheckedProduct({out_channels, channels_per_group, kernel_h, kernel_w});

    const int64_t expected_h = OutputExtent(height, pads_[0], pads_[2], kernel_h, dilations_[0], strides_[0]);
    const int64_t expected_w = OutputExtent(width, pads_[1], pads_[3], kernel_w, dilations_[1], strides_[1]);
    if (exact_shape[2] != expected_h || exact_shape[3] != expected_w) {
      throw Ort::Exception("ApproxConvCudaS8Lut exact-base shape does not match Conv attributes", ORT_INVALID_ARGUMENT);
    }

    OrtValue* output_value = nullptr;
    Throw(api_->KernelContext_GetOutput(
        context, 0, exact_shape.data(), exact_shape.size(), &output_value));
    const int8_t* x = Data<int8_t>(x_value);
    const int8_t* w = Data<int8_t>(w_value);
    const float* exact = Data<float>(exact_value);
    const int32_t* lut = Data<int32_t>(lut_value);
    float* output = MutableData<float>(output_value);
    void* raw_stream = nullptr;
    Throw(api_->KernelContext_GetGPUComputeStream(context, &raw_stream));
    if (raw_stream == nullptr) {
      throw Ort::Exception("ApproxConvCudaS8Lut could not obtain the ORT CUDA stream", ORT_RUNTIME_EXCEPTION);
    }
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    EnsureWeightQuantizationArrays(stream);

    const int64_t output_elements = CheckedProduct({n, out_channels, expected_h, expected_w});
    const int64_t int32_max = std::numeric_limits<int32_t>::max();
    if (x_elements > int32_max || w_elements > int32_max || output_elements > int32_max ||
        n > 65535 || out_channels > 65535 || expected_h * expected_w > int32_max) {
      throw Ort::Exception(
          "ApproxConvCudaS8Lut optimized CUDA indexing requires tensors within int32/grid limits",
          ORT_INVALID_ARGUMENT);
    }
    const bool int32_accumulator =
        residual_abs_bound_ >= 0 &&
        residual_terms <= int32_max / std::max<int64_t>(residual_abs_bound_, 1);
    constexpr int block_size = 256;
    const int spatial = static_cast<int>(expected_h * expected_w);
    const dim3 grid(
        static_cast<unsigned int>((spatial + block_size - 1) / block_size),
        static_cast<unsigned int>(out_channels),
        static_cast<unsigned int>(n));
#define CNN_LAUNCH_RESIDUAL_SHAPE(AccumulatorType, KernelH, KernelW)                        \
    AddSignedResidualConvSpatial<AccumulatorType, KernelH, KernelW>                         \
        <<<grid, block_size, 0, stream>>>(                                                  \
        x, w, exact, lut, output, static_cast<int>(channels), static_cast<int>(height),      \
        static_cast<int>(width), static_cast<int>(out_channels),                            \
        static_cast<int>(channels_per_group), static_cast<int>(kernel_h),                   \
        static_cast<int>(kernel_w), static_cast<int>(expected_h),                           \
        static_cast<int>(expected_w), static_cast<int>(strides_[0]),                        \
        static_cast<int>(strides_[1]), static_cast<int>(dilations_[0]),                     \
        static_cast<int>(dilations_[1]), static_cast<int>(pads_[0]),                        \
        static_cast<int>(pads_[1]), static_cast<int>(group_), scale_x_,                     \
        static_cast<int32_t>(zp_x_), device_weight_scales_, scale_w_,                       \
        device_weight_zero_points_, static_cast<int32_t>(zp_w_))
#define CNN_LAUNCH_RESIDUAL(AccumulatorType)                                                \
    do {                                                                                    \
      if (kernel_h == 1 && kernel_w == 1) {                                                 \
        CNN_LAUNCH_RESIDUAL_SHAPE(AccumulatorType, 1, 1);                                  \
      } else if (kernel_h == 3 && kernel_w == 3) {                                          \
        CNN_LAUNCH_RESIDUAL_SHAPE(AccumulatorType, 3, 3);                                  \
      } else {                                                                              \
        CNN_LAUNCH_RESIDUAL_SHAPE(AccumulatorType, 0, 0);                                  \
      }                                                                                     \
    } while (false)
    if (int32_accumulator) {
      CNN_LAUNCH_RESIDUAL(int32_t);
    } else {
      CNN_LAUNCH_RESIDUAL(int64_t);
    }
#undef CNN_LAUNCH_RESIDUAL
#undef CNN_LAUNCH_RESIDUAL_SHAPE
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
      throw Ort::Exception(cudaGetErrorString(launch_error), ORT_RUNTIME_EXCEPTION);
    }
  }

 private:
  void Throw(OrtStatus* status) const {
    if (status == nullptr) return;
    const char* raw_message = api_->GetErrorMessage(status);
    const std::string message = raw_message == nullptr ? "ONNX Runtime API call failed" : raw_message;
    api_->ReleaseStatus(status);
    throw Ort::Exception(message.c_str(), ORT_RUNTIME_EXCEPTION);
  }

  const OrtValue* Input(OrtKernelContext* context, size_t index) const {
    const OrtValue* value = nullptr;
    Throw(api_->KernelContext_GetInput(context, index, &value));
    return value;
  }

  std::vector<int64_t> Shape(const OrtValue* value) const {
    OrtTensorTypeAndShapeInfo* raw_info = nullptr;
    Throw(api_->GetTensorTypeAndShape(value, &raw_info));
    size_t rank = 0;
    OrtStatus* status = api_->GetDimensionsCount(raw_info, &rank);
    if (status != nullptr) {
      api_->ReleaseTensorTypeAndShapeInfo(raw_info);
      Throw(status);
    }
    std::vector<int64_t> shape(rank);
    status = api_->GetDimensions(raw_info, shape.data(), rank);
    api_->ReleaseTensorTypeAndShapeInfo(raw_info);
    Throw(status);
    return shape;
  }

  template <typename T>
  const T* Data(const OrtValue* value) const {
    void* raw = nullptr;
    Throw(api_->GetTensorMutableData(const_cast<OrtValue*>(value), &raw));
    return static_cast<const T*>(raw);
  }

  template <typename T>
  T* MutableData(OrtValue* value) const {
    void* raw = nullptr;
    Throw(api_->GetTensorMutableData(value, &raw));
    return static_cast<T*>(raw);
  }

  float GetFloat(const OrtKernelInfo* info, const char* name) const {
    float value = 0.0f;
    Throw(api_->KernelInfoGetAttribute_float(info, name, &value));
    return value;
  }

  int64_t GetInt(const OrtKernelInfo* info, const char* name) const {
    int64_t value = 0;
    Throw(api_->KernelInfoGetAttribute_int64(info, name, &value));
    return value;
  }

  int64_t GetIntOr(const OrtKernelInfo* info, const char* name, int64_t fallback) const {
    int64_t value = fallback;
    OrtStatus* status = api_->KernelInfoGetAttribute_int64(info, name, &value);
    if (status != nullptr) api_->ReleaseStatus(status);
    return status == nullptr ? value : fallback;
  }

  std::string GetString(const OrtKernelInfo* info, const char* name) const {
    size_t size = 0;
    Throw(api_->KernelInfoGetAttribute_string(info, name, nullptr, &size));
    std::string value(size, '\0');
    Throw(api_->KernelInfoGetAttribute_string(info, name, value.data(), &size));
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
  }

  std::vector<int64_t> GetInts(
      const OrtKernelInfo* info, const char* name, const std::vector<int64_t>& fallback) const {
    size_t size = 0;
    OrtStatus* status = api_->KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);
    if (status != nullptr) {
      api_->ReleaseStatus(status);
      return fallback;
    }
    std::vector<int64_t> values(size);
    Throw(api_->KernelInfoGetAttributeArray_int64(info, name, values.data(), &size));
    return values;
  }

  std::vector<float> GetFloats(
      const OrtKernelInfo* info, const char* name, const std::vector<float>& fallback) const {
    size_t size = 0;
    OrtStatus* status = api_->KernelInfoGetAttributeArray_float(info, name, nullptr, &size);
    if (status != nullptr) {
      api_->ReleaseStatus(status);
      return fallback;
    }
    std::vector<float> values(size);
    Throw(api_->KernelInfoGetAttributeArray_float(info, name, values.data(), &size));
    return values;
  }

  static int64_t OutputExtent(
      int64_t input, int64_t pad_before, int64_t pad_after,
      int64_t kernel, int64_t dilation, int64_t stride) {
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;
    const int64_t numerator = input + pad_before + pad_after - effective_kernel;
    if (numerator < 0) {
      throw Ort::Exception("ApproxConvCudaS8Lut Conv kernel exceeds padded input", ORT_INVALID_ARGUMENT);
    }
    return numerator / stride + 1;
  }

  static int64_t CheckedProduct(std::initializer_list<int64_t> factors) {
    int64_t product = 1;
    for (const int64_t factor : factors) {
      if (factor <= 0 || product > std::numeric_limits<int64_t>::max() / factor) {
        throw Ort::Exception("ApproxConvCudaS8Lut tensor dimensions overflow int64", ORT_INVALID_ARGUMENT);
      }
      product *= factor;
    }
    return product;
  }

  static int CurrentDeviceNoThrow() {
    int device = -1;
    return cudaGetDevice(&device) == cudaSuccess ? device : -1;
  }

  void EnsureWeightQuantizationArrays(cudaStream_t stream) {
    if ((scale_w_arr_.empty() || device_weight_scales_ != nullptr) &&
        (weight_zero_points_.empty() || device_weight_zero_points_ != nullptr)) return;
    int device = -1;
    const cudaError_t device_error = cudaGetDevice(&device);
    if (device_error != cudaSuccess) {
      throw Ort::Exception(cudaGetErrorString(device_error), ORT_RUNTIME_EXCEPTION);
    }
    cudaError_t error = cudaSuccess;
    if (!scale_w_arr_.empty() && device_weight_scales_ == nullptr) {
      const size_t bytes = scale_w_arr_.size() * sizeof(float);
      error = cudaMalloc(&device_weight_scales_, bytes);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            device_weight_scales_, scale_w_arr_.data(), bytes, cudaMemcpyHostToDevice, stream);
      }
    }
    if (error == cudaSuccess && !weight_zero_points_.empty() &&
        device_weight_zero_points_ == nullptr) {
      const size_t bytes = weight_zero_points_.size() * sizeof(int32_t);
      error = cudaMalloc(&device_weight_zero_points_, bytes);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            device_weight_zero_points_, weight_zero_points_.data(), bytes,
            cudaMemcpyHostToDevice, stream);
      }
    }
    if (error != cudaSuccess) {
      if (device_weight_scales_ != nullptr) cudaFree(device_weight_scales_);
      if (device_weight_zero_points_ != nullptr) cudaFree(device_weight_zero_points_);
      device_weight_scales_ = nullptr;
      device_weight_zero_points_ = nullptr;
      throw Ort::Exception(cudaGetErrorString(error), ORT_RUNTIME_EXCEPTION);
    }
    scale_device_ = device;
  }

  const OrtApi* api_;
  float scale_x_ = 0.0f;
  float scale_w_ = 0.0f;
  int64_t zp_x_ = 0;
  int64_t zp_w_ = 0;
  int64_t semantic_zero_mask_ = 0;
  int64_t group_ = 1;
  int64_t residual_abs_bound_ = -1;
  std::string lut_domain_;
  std::string index_encoding_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  std::vector<int64_t> pads_;
  std::vector<float> scale_w_arr_;
  std::vector<int32_t> weight_zero_points_;
  float* device_weight_scales_ = nullptr;
  int32_t* device_weight_zero_points_ = nullptr;
  int scale_device_ = -1;
};

class SignedResidualConvOp
    : public Ort::CustomOpBase<SignedResidualConvOp, SignedResidualConvKernel> {
 public:
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SignedResidualConvKernel(api, info);
  }
  const char* GetName() const { return "ApproxConvCudaS8Lut"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      case 2:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      case 3:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

static SignedResidualConvOp g_signed_residual_conv_op;

}  // namespace cnn_s8s8

extern "C" CNN_ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  OrtCustomOpDomain* domain = nullptr;
  OrtStatus* status = api->CreateCustomOpDomain("approxdomain", &domain);
  if (status != nullptr) return status;
  status = api->CustomOpDomain_Add(
      domain, static_cast<const OrtCustomOp*>(&cnn_s8s8::g_signed_residual_conv_op));
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
