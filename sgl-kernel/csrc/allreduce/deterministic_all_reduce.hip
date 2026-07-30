// Deterministic All-Reduce for ROCm/HIP
//
// This wrapper drives the existing cross-device reduce kernels, both of which
// are deterministic because they accumulate each output element in a fixed
// absolute peer order (0..ngpus-1), independent of rank and tensor size (no
// atomics, no race conditions):
//
//   - cross_device_reduce_1stage: every GPU reads ALL peers' data and reduces
//     the whole tensor locally in fixed order. Lowest latency for small sizes.
//   - cross_device_reduce_2stage: reduce-scatter (each GPU reduces only its
//     1/ngpus partition, in the same fixed absolute order) followed by an
//     all-gather of the partitions. Much faster for large sizes since each GPU
//     reduces 1/ngpus of the data instead of the whole tensor.
//
// Both produce bitwise-identical results, so we pick between them purely by
// size (see det_use_2stage), mirroring CustomAllreduce::allreduce().

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "custom_all_reduce_hip.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// Helper function for weak contiguity check
bool _is_weak_contiguous_det(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

// Both cross_device_reduce_1stage and cross_device_reduce_2stage accumulate
// each element in a fixed absolute peer order (0..ngpus-1), independent of rank
// and tensor size, so both are deterministic. Pick the 2-stage (reduce-scatter
// + all-gather) kernel for larger buffers where it is much faster -- each rank
// only reduces 1/ngpus of the data instead of the whole tensor -- and fall back
// to 1-stage for small buffers and world_size == 2, mirroring the heuristic in
// CustomAllreduce::allreduce().
static inline bool det_use_2stage(int world_size, size_t bytes) {
  if (world_size <= 2) return false;
  bool small = (world_size <= 4 && bytes < 512 * 1024) || (world_size <= 8 && bytes < 256 * 1024);
  return !small;
}

#define DET_KL(ngpus, KERNEL)                                                                    \
  hipLaunchKernelGGL((sglang::KERNEL<T, ngpus>), dim3(blocks), dim3(threads), 0, stream, ptrs,   \
                     fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size)

#define DET_DISPATCH()                                                    \
  do {                                                                    \
    if (use_2stage) {                                                     \
      switch (fa->world_size_) {                                          \
        case 2: DET_KL(2, cross_device_reduce_2stage); break;             \
        case 4: DET_KL(4, cross_device_reduce_2stage); break;             \
        case 6: DET_KL(6, cross_device_reduce_2stage); break;             \
        case 8: DET_KL(8, cross_device_reduce_2stage); break;             \
        default: throw std::runtime_error("world_size must be in (2,4,6,8)"); \
      }                                                                   \
    } else {                                                              \
      switch (fa->world_size_) {                                          \
        case 2: DET_KL(2, cross_device_reduce_1stage); break;             \
        case 4: DET_KL(4, cross_device_reduce_1stage); break;             \
        case 6: DET_KL(6, cross_device_reduce_1stage); break;             \
        case 8: DET_KL(8, cross_device_reduce_1stage); break;             \
        default: throw std::runtime_error("world_size must be in (2,4,6,8)"); \
      }                                                                   \
    }                                                                     \
  } while (0)

// Deterministic all-reduce for registered buffers (ROCm)
void deterministic_all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous_det(out));
  TORCH_CHECK(_is_weak_contiguous_det(inp));

  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);

  // For ROCm, manually call the 1-stage kernel to ensure deterministic ordering
  // Get rank data pointer
  sglang::RankData* ptrs;
  hipStreamCaptureStatus status;
  AT_CUDA_CHECK(hipStreamIsCapturing(stream, &status));
  if (status == hipStreamCaptureStatusActive) {
    ptrs = fa->d_rank_data_base_ + fa->graph_unreg_buffers_.size();
    fa->graph_unreg_buffers_.push_back(inp.data_ptr());
  } else {
    auto it = fa->buffers_.find(inp.data_ptr());
    if (it == fa->buffers_.end()) {
      throw std::runtime_error("buffer not registered!");
    }
    ptrs = it->second;
  }

  int size = out.numel();
  int threads = 512;

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      using T = float;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      bool use_2stage = det_use_2stage(fa->world_size_, size * sizeof(P));
      DET_DISPATCH();
      break;
    }
    case at::ScalarType::Half: {
      using T = half;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      bool use_2stage = det_use_2stage(fa->world_size_, size * sizeof(P));
      DET_DISPATCH();
      break;
    }
#if (__HIP_ARCH__ >= 800 || !defined(__HIP_ARCH__))
    case at::ScalarType::BFloat16: {
      using T = nv_bfloat16;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      bool use_2stage = det_use_2stage(fa->world_size_, size * sizeof(P));
      DET_DISPATCH();
      break;
    }
#endif
    default:
      throw std::runtime_error("deterministic allreduce only supports float32, float16 and bfloat16");
  }
}

#undef DET_DISPATCH
#undef DET_KL

// Deterministic all-reduce for unregistered buffers (ROCm)
void deterministic_all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer, torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

  auto input_size = inp.numel() * inp.element_size();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
              "registered buffer is too small to contain the input");
  AT_CUDA_CHECK(hipMemcpyAsync(reg_buffer.data_ptr(), inp.data_ptr(),
                                input_size, hipMemcpyDeviceToDevice, stream));
  deterministic_all_reduce_reg(_fa, reg_buffer, out);
}
