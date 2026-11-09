// Custom all-reduce kernels over the decoupled Communicator storage plane.
//
// Five algorithms are provided behind one entry point:
//   - 1shot_push: lamport-style push of local data to every peer's push
//     workspace, then a local polling reduce (best at small sizes).
//   - 2shot_push: the same lamport push, but each chunk goes only to its
//     owner rank, which reduces it and pushes the sum to every other rank.
//     Same traffic as 2shot_pull, with no barriers and no remote loads.
//   - 1shot_pull: every rank reduces all peers' data (from the symmetric pull
//     workspaces, a CUDA-graph pointer table, or a multicast address).
//   - 2shot_pull: reduce-scatter fused with all-gather; each rank reduces its
//     shard in place so every workspace ends up holding the full result.
//   - ring_simple (ROCm gfx1250 only): a ring all-reduce; each chunk goes
//     to the next rank only, through a fixed-size FIFO behind
//     64-bit counters, in 2 * (world_size - 1) hops per loop.
//
// Unlike the previous implementation, the kernels carry no storage or IPC
// logic: all pointers arrive via the communication planes (owned by Python)
// and the per-call params. The push and pull families take disjoint params,
// so neither carries the other's pointer table into its grid constants.
//
// The pull family reduces over the pull plane's workspaces, which exist
// because this kernel's callers hand it plain tensors: the host stages the
// input in and copies the result back out. Callers that already allocate
// from symmetric memory (the K3 fused paths) reduce in place instead and
// borrow the plane only to barrier on.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

// Kept out of sgl_kernel/utils.cuh: every JIT build's cache key covers that
// header's contents, so an edit there rebuilds every CUDA kernel.
#ifdef USE_ROCM
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaGetDevice hipGetDevice
#endif

namespace sglang {

using device::distributed::PullWorkSpace, device::distributed::PushWorkSpace, device::distributed::RingWorkSpace;
using host::distributed::CommunicatorRef;

inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;

template <uint32_t kWorldSize>
struct AllReducePushParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  PushWorkSpace<kWorldSize> ws;
};

struct AllReduceRingParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  RingWorkSpace ws;
};

template <uint32_t kWorldSize>
struct AllReducePullParams {
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  void* const* __restrict__ graph_params;
  PullWorkSpace<kWorldSize> ws;
};

/// `vec_offset` is a *vector index* and must stay folded into the base pointers
/// here: a typed 32-bit bias becomes one widening multiply-add off a
/// constant-bank base, which ptxas keeps on the uniform datapath, so the whole
/// peer table lives in uniform registers. Biasing `vid` at each access instead,
/// or using a 64-bit byte bias (the uniform datapath has no 64-bit add), makes
/// all `kWorldSize` addresses thread-varying and costs 2shot ~12 registers per
/// thread and ~10% throughput.
template <typename V, uint32_t kWorldSize, bool kUseGraph>
struct LoadStoreImpl {
 public:
  static constexpr uint32_t size() {
    return kWorldSize;
  }
  SGL_DEVICE LoadStoreImpl(const AllReducePullParams<kWorldSize>& params, uint32_t vec_offset = 0) {
    if constexpr (kUseGraph) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        m_data[i] = reinterpret_cast<V*>(params.graph_params[i]) + vec_offset;
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        m_data[i] = reinterpret_cast<V*>(params.ws.workspaces[i]) + vec_offset;
      }
    }
  }

  SGL_DEVICE void load_reduce(V& vec, uint32_t vid) const {
    V vecs[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      vecs[i].load(m_data[i], vid);
    }
    vec = device::reduce_vec(vecs);
  }

  SGL_DEVICE void store_multi(const V& val, uint32_t vid) const {
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      val.store(m_data[i], vid);
    }
  }

 private:
  V* m_data[kWorldSize];
};

template <typename V, uint32_t kWorldSize, bool kUseGraph>
struct MultiCastImpl {
 public:
  static_assert(kUseGraph == false);
  static constexpr uint32_t size() {
    return kWorldSize;
  }

  SGL_DEVICE MultiCastImpl(const AllReducePullParams<kWorldSize>& params, uint32_t vec_offset = 0)
      : m_multicast_ptr(reinterpret_cast<V*>(params.ws.mc_workspace) + vec_offset) {}

  SGL_DEVICE void load_reduce(V& vec, uint32_t vid) const {
    device::ptx::ld_multimem_16B(vec, m_multicast_ptr, vid);
  }

  SGL_DEVICE void store_multi(const V& val, uint32_t vid) const {
    return device::ptx::st_multimem_16B(val, m_multicast_ptr, vid);
  }

 private:
  V* m_multicast_ptr;
};

#define ALL_REDUCE_KERNEL __global__ __launch_bounds__(1024, 1)

template <typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_1shot_push_kernel(const __grid_constant__ AllReducePushParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto num_threads = blockDim.x * gridDim.x;
  // Round-robin warps to blocks rather than giving each block a contiguous run.
  // The grid is pinned to the counter array, so once `num_vecs` stops filling
  // `gridDim * blockDim` a block-major index leaves the tail CTAs with nothing
  // to do; that happens over a whole 2x band of sizes, between the point where
  // `choose_block_size` gives up on 512 and the point where 1024 threads fill
  // the grid again.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // push to peer
  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }

  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    vec.load(params.input, vid);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
    }
  }

  // poll from local
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce_vec(vec);
    out_vec.store(params.output, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

/// Chunk c of the flat tensor is owned by rank c. A slot (dst, src) holds two
/// `chunk_vecs` halves: [0, chunk) takes src's partial of chunk dst, and
/// [chunk, 2 * chunk) takes src's reduced chunk src. The (r, r) slot is unused:
/// the owner reads its own partial straight from the input.
template <typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_2shot_push_kernel(const __grid_constant__ AllReducePushParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kFan = kWorldSize - 1;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto chunk_vecs = div_ceil(num_vecs, kWorldSize);
  const auto get_chunk_len = [&](uint32_t c) {  //
    return min(chunk_vecs, num_vecs - min(num_vecs, c * chunk_vecs));
  };
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_warps = num_threads / kWarpThreads;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;

  // peer will cover 0..kWorldSize - 1, except for local rank `r`
  const auto peer = (r + 1 + global_warp_id % kFan) % kWorldSize;
  const auto group_warps = num_warps / kFan + (global_warp_id % kFan < num_warps % kFan ? 1 : 0);
  const auto group_tid = (global_warp_id / kFan) * kWarpThreads + lane_id;
  const auto group_stride = group_warps * kWarpThreads;
  const auto peer_len = get_chunk_len(peer);

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of the peer's chunk to its owner
  {
    const auto push_ptr = epoch.slot_ptr(/*dst=*/peer, /*src=*/r);
    for (auto j = group_tid; j < peer_len; j += group_stride) {
      vec_t vec;
      vec.load(params.input, peer * chunk_vecs + j);
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, push_ptr, j);
    }
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }

  // reduce my chunk, keep it and fan it out
  const auto my_len = get_chunk_len(r);
  for (auto j = global_tid; j < my_len; j += num_threads) {
    const auto vid = r * chunk_vecs + j;
    vec_t vec[kWorldSize];
    vec_t local_vec;
    local_vec.load(params.input, vid);
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= ((i != r) & Lamport::has_pos_zero(vec[i].data()));
      }
      if (!has_zero) break;
    } while (true);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (i == r) vec[i] = local_vec;
    }
    auto out_vec = reduce_vec(vec);
    // remap before the local store too, so every rank holds identical bits
    Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (i != r) ptx::st_relaxed_16B(out_vec, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
    }
    out_vec.store(params.output, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (i != r) ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], j);
    }
  }

  // all-gather: drain the peer's reduced chunk
  {
    const auto poll_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/peer);
    for (auto j = group_tid; j < peer_len; j += group_stride) {
      vec_t vec;
      do {
        ptx::ld_relaxed_16B(vec, poll_ptr, chunk_vecs + j);
      } while (Lamport::has_pos_zero(vec.data()));
      vec.store(params.output, peer * chunk_vecs + j);
      ptx::st_global_16B(pos_zero_vec, poll_ptr, chunk_vecs + j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

#ifdef USE_ROCM
/// One half of a 2-rank 2shot_push, for the pair stages of the DPX hierarchical
/// all-reduce. Chunk c of the full tensor belongs to rank c.
///   reduce-scatter: `input` is the full tensor, `output` this rank's reduced
///                   chunk (`params.num_vecs` counts the full tensor).
///   all-gather:     `input` is this rank's chunk, `output` the full tensor.
/// Same Lamport markers and epochs as 2shot_push, with a one-chunk slot: each
/// launch both pushes to and polls (and clears) the peer, so the two stages
/// alternate epochs in step on both ranks and may share the slot.
template <typename T, bool kAllGather, bool kUsePDL>
ALL_REDUCE_KERNEL void pair_push_kernel(const __grid_constant__ AllReducePushParams<2> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto peer = r ^ 1;
  const auto chunk_vecs = params.num_vecs / 2;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<2>{params.ws};
  const auto push_ptr = epoch.slot_ptr(/*dst=*/peer, /*src=*/r);
  const auto poll_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/peer);

  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    vec_t vec;
    vec.load(params.input, kAllGather ? j : peer * chunk_vecs + j);
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, push_ptr, j);
    // remap before the local store too, so both ranks hold identical bits
    if constexpr (kAllGather) vec.store(params.output, r * chunk_vecs + j);
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    // vec[0] arrives from the peer, vec[1] is local: a two-term sum is the same
    // in either order
    vec_t vec[2];
    do {
      ptx::ld_relaxed_16B(vec[0], poll_ptr, j);
    } while (Lamport::has_pos_zero(vec[0].data()));
    if constexpr (kAllGather) {
      vec[0].store(params.output, peer * chunk_vecs + j);
    } else {
      vec[1].load(params.input, r * chunk_vecs + j);
      reduce_vec(vec).store(params.output, j);
    }
    ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

/// Threads per ring_simple channel, and 16-byte packs each thread keeps in
/// flight per pass.
inline constexpr uint32_t kRingSimpleThreads = 256;
inline constexpr uint32_t kRingSimpleUnroll = 32;

/// `vecs[u]` to `base` + u * kStride vectors as streaming system-scope stores,
/// eight per `st_relaxed_nt_x8_16B`.
template <uint32_t kStride, typename V, uint32_t... G>
SGL_DEVICE void st_relaxed_nt_run(const V* vecs, void* base, std::integer_sequence<uint32_t, G...>) {
  constexpr auto kStep = static_cast<int32_t>(kStride * sizeof(V));
  (device::ptx::st_relaxed_nt_x8_16B<static_cast<int32_t>(G * 8) * kStep, kStep>(vecs + G * 8, base), ...);
}

/// What one ring hop does with each vector: receive it, add this rank's
/// input, store it to the output, send it on.
template <bool kRecv_, bool kReduce_, bool kStoreOut_, bool kSend_>
struct RingHop {
  static constexpr bool kRecv = kRecv_, kReduce = kReduce_, kStoreOut = kStoreOut_, kSend = kSend_;
};

/// The ring all-reduce: one channel per block,
/// a FIFO of `num_slots` chunk-sized slices per channel in the next rank's
/// memory, and per channel three 64-bit counters:
/// `ready` (slices the predecessor published here), `drained` (slices the
/// successor consumed) and `packet` (this rank's own count). A slice is plain
/// data, published once all its stores have completed, so the receiver polls
/// one counter and reads the payload once. Payload goes out as streaming
/// system-scope stores, the output as streaming stores.
///
/// A channel is one block of 8 waves, at most 2 per SIMD; `amdgpu_waves_per_eu`
/// says so, or the compiler budgets registers for more.
template <typename T, uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kRingSimpleThreads, 1) __attribute__((
    amdgpu_waves_per_eu(1, 2))) void all_reduce_ring_simple_kernel(const __grid_constant__ AllReduceRingParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kNumPackets = 2 * (kWorldSize - 1);
  constexpr uint32_t kWarpVecs = kWarpThreads * kRingSimpleUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto& ws = params.ws;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto ch = blockIdx.x;
  const auto num_channels = gridDim.x;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto num_warps = blockDim.x / kWarpThreads;
  const auto slot_mask = ws.num_slots - 1;
  const auto slot_vecs = ws.slot_bytes / 16;
  const auto fifo_offset = ch * ws.num_slots * ws.slot_bytes;
  const auto recv_fifo = ws.recv + fifo_offset;
  const auto send_fifo = ws.send + fifo_offset;
  auto& sem = ws.credits[ch];
  auto& prev_sem = ws.prev_credits[ch];
  auto& next_sem = ws.next_credits[ch];
  const auto slot_of = [&](uint8_t* fifo, uint64_t p) { return fifo + (p & slot_mask) * ws.slot_bytes; };

  PDLWaitPrimary<kUsePDL>();
  uint64_t packet = sem.packet;
  // polled by thread 0 only; the barrier after each wait publishes the result
  uint64_t ready = 0, drained = 0;
  const auto wait_until = [&](uint64_t& seen, const uint64_t* ptr, uint64_t target) {
    while (seen < target) {
      seen = ptx::load_acquire_sys(ptr);
      if (seen < target) __builtin_amdgcn_s_sleep(1);
    }
  };
  // Packet p may be received once the predecessor published it, and sent
  // once the successor drained the packet that last held its slot.
  const auto wait = [&](bool recv, uint64_t recv_p, bool send, uint64_t send_p) {
    if (threadIdx.x == 0) {
      if (recv) wait_until(ready, &sem.ready, recv_p + 1);
      if (send && send_p >= ws.num_slots) wait_until(drained, &sem.drained, send_p + 1 - ws.num_slots);
    }
    __syncthreads();
  };
  // Once every wave's stores (and so the loads feeding them) completed: tell
  // the predecessor `recv_p` is drained and the successor `send_p` is ready.
  const auto post = [&](bool recv, uint64_t recv_p, bool send, uint64_t send_p) {
    ptx::wait_stores_sys();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (recv) ptx::store_relaxed_sys(&prev_sem.drained, recv_p + 1);
      if (send) ptx::store_relaxed_sys(&next_sem.ready, send_p + 1);
    }
  };
  // One slice: warps take contiguous
  // hunks of kWarpVecs, each lane a pack per kWarpThreads vectors, so a lane
  // has kRingSimpleUnroll loads per source in flight. Only whole hunks take the
  // unrolled path: a bounds check per pack splits it into branches the
  // compiler fences with a full load wait each, serializing the loads.
  const auto copy_packs =
      [&](auto unroll, auto mode, const uint8_t* recv_slot, uint8_t* send_slot, uint32_t start, uint32_t j) {
        constexpr uint32_t kUnroll = decltype(unroll)::value;
        constexpr uint32_t kStride = kUnroll == 1 ? 1 : kWarpThreads;
        constexpr bool kRecv = decltype(mode)::kRecv, kReduce = decltype(mode)::kReduce;
        constexpr bool kStoreOut = decltype(mode)::kStoreOut, kSend = decltype(mode)::kSend;
        // Per-lane base pointers with constant pack offsets, which fold into the
        // instructions' immediate offsets: an index recomputed per pack reuses
        // its VGPR, and each store then waits (s_wait_xcnt) for the last to read it.
        const auto in = static_cast<const vec_t*>(params.input) + start + j;
        const auto out = static_cast<vec_t*>(params.output) + start + j;
        const auto recv_at = reinterpret_cast<const vec_t*>(recv_slot) + j;
        const auto send_at = reinterpret_cast<vec_t*>(send_slot) + j;
        vec_t vecs[kUnroll];
        vec_t inputs[kUnroll];
    // every pack of the local input, then every pack of the FIFO
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (!kRecv || kReduce) inputs[u] = in[u * kStride];
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (kRecv) ptx::ld_relaxed_16B(vecs[u], recv_at, u * kStride);
        }
        // keep every load ahead of the first use: left alone, the scheduler pulls
        // each reduction up behind its two loads and waits there, to save registers
        __builtin_amdgcn_sched_barrier(0);
    // results land in `vecs`; then the stores, one destination at a time
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (kRecv && kReduce) {
            vec_t pair[2];
            pair[0] = vecs[u];
            pair[1] = inputs[u];
            vecs[u] = reduce_vec(pair);
          } else if constexpr (!kRecv) {
            vecs[u] = inputs[u];
          }
        }
        if constexpr (kStoreOut) {
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u)
            ptx::st_nontemporal_16B(vecs[u], out, u * kStride);
        }
        if constexpr (kSend) {
          if constexpr (kUnroll % 8 == 0) {
            st_relaxed_nt_run<kStride>(vecs, send_at, std::make_integer_sequence<uint32_t, kUnroll / 8>{});
          } else {
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u)
              ptx::st_relaxed_16B(vecs[u], send_at, u * kStride);
          }
        }
      };
  const auto copy = [&](auto mode, const uint8_t* recv_slot, uint8_t* send_slot, uint32_t start, uint32_t n) {
    const auto full = n / kWarpVecs * kWarpVecs;
    for (uint32_t base = warp_id * kWarpVecs; base < full; base += num_warps * kWarpVecs) {
      copy_packs(
          std::integral_constant<uint32_t, kRingSimpleUnroll>{}, mode, recv_slot, send_slot, start, base + lane_id);
    }
    for (uint32_t j = full + threadIdx.x; j < n; j += blockDim.x) {
      copy_packs(std::integral_constant<uint32_t, 1>{}, mode, recv_slot, send_slot, start, j);
    }
  };

  // each channel takes a contiguous share of the tensor, then loops over it in
  // kWorldSize chunks of at most one slot
  const auto channel_vecs = div_ceil(num_vecs, num_channels);
  const auto channel_start = min(ch * channel_vecs, num_vecs);
  const auto channel_len = min(channel_vecs, num_vecs - channel_start);
  const auto loop_vecs = kWorldSize * slot_vecs;
  for (uint32_t base = 0; base < channel_len; base += loop_vecs) {
    const auto len = min(loop_vecs, channel_len - base);
    const auto chunk_vecs = div_ceil(len, kWorldSize);
    const auto chunk_start = [&](uint32_t c) { return channel_start + base + c * chunk_vecs; };
    const auto chunk_len = [&](uint32_t c) { return min(chunk_vecs, len - min(len, c * chunk_vecs)); };

    // first hop: this rank's own chunk
    wait(false, 0, true, packet);
    copy(RingHop<false, false, false, true>{}, nullptr, slot_of(send_fifo, packet), chunk_start(r), chunk_len(r));
    post(false, 0, true, packet);

    // hop q receives chunk r - 1 - q: a partial for q < kWorldSize - 1, which
    // this rank completes at q = kWorldSize - 2, then a reduced chunk
    for (uint32_t q = 0; q < kNumPackets; ++q, ++packet) {
      const bool forward = q + 1 < kNumPackets;
      const auto c = (r + 2 * kWorldSize - 1 - q) % kWorldSize;
      const auto recv_slot = slot_of(recv_fifo, packet);
      const auto send_slot = slot_of(send_fifo, packet + 1);
      const auto start = chunk_start(c);
      const auto n = chunk_len(c);
      wait(true, packet, forward, packet + 1);
      if (q + 2 < kWorldSize) {
        copy(RingHop<true, true, false, true>{}, recv_slot, send_slot, start, n);
      } else if (q + 2 == kWorldSize) {
        copy(RingHop<true, true, true, true>{}, recv_slot, send_slot, start, n);
      } else if (forward) {
        copy(RingHop<true, false, true, true>{}, recv_slot, send_slot, start, n);
      } else {
        copy(RingHop<true, false, true, false>{}, recv_slot, send_slot, start, n);
      }
      post(true, packet, forward, packet + 1);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  if (threadIdx.x == 0) sem.packet = packet;
}

/// Threads per unrolled 2shot_push block, and 16-byte packs each lane keeps in
/// flight in its scatter and gather phases. The reduce phase holds a pack from
/// every rank per unroll step, so it unrolls less as the world grows.
inline constexpr uint32_t kTwoShotPushThreads = 256;
inline constexpr uint32_t kTwoShotPushUnroll = 8;
template <uint32_t kWorldSize>
inline constexpr uint32_t kTwoShotPushReduceUnroll = kWorldSize <= 4 ? 8 : 4;
/// Below these sizes the per-vector kernel wins: one vector per thread spreads a
/// chunk over more warps than whole hunks do (TP2/TP4 unrolled are 10-14% slower
/// at 128 KB - 1 MB), and with one rank arriving late TP2/TP4 stay 3-10% slower
/// up to 4 MB.
template <uint32_t kWorldSize>
inline constexpr int64_t kTwoShotPushUnrolledMinBytes =
    int64_t{
        kWorldSize <= 4  ? 8
        : kWorldSize < 8 ? 2
                         : 1}
    << 20;

/// `kUnroll` vectors, `kStride` vectors apart, as streaming system-scope stores.
template <uint32_t kStride, uint32_t kUnroll, typename V>
SGL_DEVICE void st_relaxed_nt_unrolled(const V* vecs, void* base) {
  if constexpr (kUnroll % 8 == 0) {
    st_relaxed_nt_run<kStride>(vecs, base, std::make_integer_sequence<uint32_t, kUnroll / 8>{});
  } else {
    static_assert(kUnroll == 4);
    constexpr auto kStep = static_cast<int32_t>(kStride * sizeof(V));
    device::ptx::st_relaxed_nt_x4_16B<0, kStep>(vecs, base);
  }
}

/// `all_reduce_2shot_push_kernel` with the ring_simple memory pipeline: same
/// slots, markers and epochs, but warps take contiguous hunks of
/// `kWarpThreads * unroll` vectors and issue every load of a hunk before using
/// any, peer payload goes out as batched streaming system-scope stores and the
/// output as streaming stores. The tail past the last whole hunk takes the
/// per-vector path. Single node only (at most 8 ranks).
template <typename T, uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kTwoShotPushThreads, 1)
    __attribute__((amdgpu_waves_per_eu(1, 2))) void all_reduce_2shot_push_unrolled_kernel(
        const __grid_constant__ AllReducePushParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kFan = kWorldSize - 1;
  constexpr uint32_t kUnroll = kTwoShotPushUnroll;
  constexpr uint32_t kReduceUnroll = kTwoShotPushReduceUnroll<kWorldSize>;
  constexpr uint32_t kHunk = kWarpThreads * kUnroll;
  constexpr uint32_t kReduceHunk = kWarpThreads * kReduceUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto chunk_vecs = div_ceil(num_vecs, kWorldSize);
  const auto get_chunk_len = [&](uint32_t c) {  //
    return min(chunk_vecs, num_vecs - min(num_vecs, c * chunk_vecs));
  };
  const auto input = static_cast<const vec_t*>(params.input);
  const auto output = static_cast<vec_t*>(params.output);
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_warps = num_threads / kWarpThreads;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;

  // peer will cover 0..kWorldSize - 1, except for local rank `r`
  const auto peer = (r + 1 + global_warp_id % kFan) % kWorldSize;
  const auto group_warp = global_warp_id / kFan;
  const auto group_warps = num_warps / kFan + (global_warp_id % kFan < num_warps % kFan ? 1 : 0);
  const auto group_tid = group_warp * kWarpThreads + lane_id;
  const auto group_stride = group_warps * kWarpThreads;
  const auto peer_len = get_chunk_len(peer);
  const auto peer_full = peer_len / kHunk * kHunk;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of the peer's chunk to its owner
  {
    const auto src = input + peer * chunk_vecs;
    const auto push_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/peer, /*src=*/r));
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto in = src + j;
      vec_t vecs[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        vecs[u] = in[u * kWarpThreads];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        Lamport::clear_pos_zero(vecs[u].data());
      }
      st_relaxed_nt_unrolled<kWarpThreads, kUnroll>(vecs, push_ptr + j);
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec = src[j];
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, push_ptr, j);
    }
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  vec_t* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/r, /*src=*/i));
  }

  // reduce my chunk, keep it and fan it out
  {
    const auto my_len = get_chunk_len(r);
    const auto my_full = my_len / kReduceHunk * kReduceHunk;
    const auto my_in = input + r * chunk_vecs;
    const auto my_out = output + r * chunk_vecs;
    // this rank's own partial comes straight from the input, through the same
    // loads as the peers' so no branch splits the batch
    const vec_t* srcs[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      srcs[i] = i == r ? my_in : poll_ptrs[i];
    }
    for (uint32_t base = global_warp_id * kReduceHunk; base < my_full; base += num_warps * kReduceHunk) {
      const auto j = base + lane_id;
      vec_t vecs[kReduceUnroll][kWorldSize];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vecs[u][i], srcs[i] + j, u * kWarpThreads);
        }
      }
      // A late peer is re-polled alone, with a sleep per spin: re-polling every
      // rank's packs floods the memory the peers' pushes land in.
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        while (true) {
          bool has_zero = false;
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            has_zero |= ((i != r) & Lamport::has_pos_zero(vecs[u][i].data()));
          }
          if (!has_zero) break;
          __builtin_amdgcn_s_sleep(1);
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::ld_relaxed_16B(vecs[u][i], srcs[i] + j, u * kWarpThreads);
          }
        }
      }
      vec_t outs[kReduceUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        outs[u] = reduce_vec(vecs[u]);
        // remap before the local store too, so every rank holds identical bits
        Lamport::clear_pos_zero(outs[u].data());
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
          const auto send_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/i, /*src=*/r));
          st_relaxed_nt_unrolled<kWarpThreads, kReduceUnroll>(outs, send_ptr + chunk_vecs + j);
        }
      }
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        ptx::st_nontemporal_16B(outs[u], my_out + j, u * kWarpThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::st_global_16B(pos_zero_vec, poll_ptrs[i] + j, u * kWarpThreads);
          }
        }
      }
    }
    for (auto j = my_full + global_tid; j < my_len; j += num_threads) {
      vec_t vec[kWorldSize];
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vec[i], srcs[i], j);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          has_zero |= ((i != r) & Lamport::has_pos_zero(vec[i].data()));
        }
        if (!has_zero) break;
      } while (true);
      auto out_vec = reduce_vec(vec);
      Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::st_relaxed_16B(out_vec, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
      }
      out_vec.store(my_out, j);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], j);
      }
    }
  }

  // all-gather: drain the peer's reduced chunk
  {
    const auto poll_ptr = poll_ptrs[peer] + chunk_vecs;
    const auto dst = output + peer * chunk_vecs;
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto at = poll_ptr + j;
      vec_t vecs[kUnroll];
      while (true) {
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          ptx::ld_relaxed_16B(vecs[u], at, u * kWarpThreads);
        }
        bool has_zero = false;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          has_zero |= Lamport::has_pos_zero(vecs[u].data());
        }
        if (!has_zero) break;
        __builtin_amdgcn_s_sleep(1);
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        ptx::st_nontemporal_16B(vecs[u], dst + j, u * kWarpThreads);
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        ptx::st_global_16B(pos_zero_vec, at, u * kWarpThreads);
      }
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec;
      do {
        ptx::ld_relaxed_16B(vec, poll_ptr, j);
      } while (Lamport::has_pos_zero(vec.data()));
      dst[j] = vec;
      ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}
#endif

template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_1shot_pull_kernel(const __grid_constant__ AllReducePullParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_vecs = params.num_vecs;
  const auto impl = Impl{params};
  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>{
      params.ws.semaphores.data(),
      params.rank,
      /*num_arrives=*/2,
  };
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    vec.store(params.output, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_relaxed(/*n=*/1);
}

template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_2shot_pull_kernel(const __grid_constant__ AllReducePullParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_total_vecs = params.num_vecs;
  const auto avg_vecs = num_total_vecs / kWorldSize;
  const auto rem_vecs = num_total_vecs % kWorldSize;
  const auto num_vecs = avg_vecs + (params.rank < rem_vecs ? 1 : 0);
  const auto vec_offset = params.rank * avg_vecs + min(params.rank, rem_vecs);
  const auto impl = Impl{params, vec_offset};
  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>{
      params.ws.semaphores.data(),
      params.rank,
      /*num_arrives=*/2,
  };
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    impl.store_multi(vec, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
#ifdef USE_ROCM
  // s_barrier does not drain a wave's outstanding stores and the signalling
  // thread's release covers only its own wave, so each wave publishes its own.
  ptx::fence_release_sys();
#endif
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}

template <uint32_t N>
__global__ void memcpy_kernel(void* __restrict__ dst, const void* __restrict__ src, uint32_t num_vecs) {
  static_assert(N % 4 == 0, "at least 4-bytes aligned for uint32_t load/store");
  using vec_t = device::AlignedVector<uint32_t, N / 4>;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  if (tid < num_vecs) {
    vec_t vec;
    vec.load(src, tid);
    vec.store(dst, tid);
  }
}

inline auto choose_block_size(uint32_t num_threads) -> uint32_t {
  static const uint32_t kNumSM = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return host::runtime::get_sm_count(device);
  }();
  for (const uint32_t block_size : {128u, 256u, 512u}) {
    if (host::div_ceil(num_threads, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}

#ifdef USE_ROCM
inline bool is_gfx1250() {
  static const bool kIsGfx1250 = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    hipDeviceProp_t prop;
    CHECK_CUDA(hipGetDeviceProperties(&prop, device));
    return std::strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
  }();
  return kIsGfx1250;
}
#endif

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct AllReduceKernel {
 private:
  using Tensor = tvm::ffi::Tensor;
  using TensorView = tvm::ffi::TensorView;
  using PushParams = AllReducePushParams<kWorldSize>;
  using PullParams = AllReducePullParams<kWorldSize>;
  using vec_t = device::AlignedVector<packed_t<T>, 16 / (sizeof(T) * 2)>;

 public:
  static Tensor
  run(const CommunicatorRef comm_ref,
      const Tensor in,
      const std::string algo,
      const tvm::ffi::Optional<TensorView> graph_params_opt,
      const bool use_multicast) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(
        algo == "1shot_pull" || algo == "2shot_pull" || algo == "1shot_push" || algo == "2shot_push" ||
        algo == "pair_reduce_scatter" || algo == "pair_all_gather" ||
        algo == "ring_simple")
        << algo;
    CHECK_HOST(comm.get_world_size() == kWorldSize) << comm.get_world_size();
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems_int64 = in.numel();
    const auto num_elems = static_cast<uint32_t>(num_elems_int64);
    const auto nbytes = static_cast<int64_t>(num_elems_int64 * sizeof(T));
    CHECK_HOST(static_cast<int64_t>(num_elems) == num_elems_int64) << num_elems_int64;
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto num_vecs = static_cast<uint32_t>(num_elems / (16 / sizeof(T)));
    const auto stream = LaunchKernel::resolve_device(in.device());
    const auto use_graph = graph_params_opt.has_value();

    if (algo == "pair_reduce_scatter" || algo == "pair_all_gather") {
      CHECK_HOST(!use_graph && !use_multicast);
#ifdef USE_ROCM
      if constexpr (kWorldSize == 2) {
        const auto& push = comm.get_push_obj();
        const auto all_gather = algo == "pair_all_gather";
        // both kernels see the full tensor's vector count
        const auto full_vecs = all_gather ? 2 * num_vecs : num_vecs;
        CHECK_HOST(full_vecs % 2 == 0) << "a pair reduce-scatter needs an even vector count: " << full_vecs;
        const auto out_elems = all_gather ? 2 * num_elems_int64 : num_elems_int64 / 2;
        Tensor out = ffi::empty({out_elems}, in.dtype(), in.device());
        const auto params = PushParams{
            .input = in.data_ptr(),
            .output = out.data_ptr(),
            .num_vecs = full_vecs,
            .rank = push.rank,
            .ws = push.get_workspace<kWorldSize>(int64_t{full_vecs / 2} * 16),
        };
        const auto kernel = all_gather ? pair_push_kernel<T, true, kUsePDL> : pair_push_kernel<T, false, kUsePDL>;
        LaunchKernel(push.num_blocks, choose_block_size(full_vecs / 2), stream)  //
            .enable_pdl(kUsePDL)(kernel, params);
        return out;
      }
#endif
      Panic("pair stages need ROCm and a 2-rank communicator, got ", kWorldSize, " ranks");
    }

    if (algo == "ring_simple") {
      CHECK_HOST(!use_graph && !use_multicast);
#ifdef USE_ROCM
      CHECK_HOST(is_gfx1250()) << "The ring all-reduce is only enabled on gfx1250";
      const auto& ring = comm.get_ring_simple_obj();
      Tensor out = ffi::empty_like(in);
      const auto params = AllReduceRingParams{
          .input = in.data_ptr(),
          .output = out.data_ptr(),
          .num_vecs = num_vecs,
          .rank = ring.rank,
          .ws = ring.get_workspace(),
      };
      LaunchKernel(ring.num_rings, kRingSimpleThreads, stream)  //
          .enable_pdl(kUsePDL)(all_reduce_ring_simple_kernel<T, kWorldSize, kUsePDL>, params);
      return out;
#else
      Panic("The ring all-reduce is only enabled on ROCm gfx1250");
#endif
    }

    if (algo == "1shot_push" || algo == "2shot_push") {
      CHECK_HOST(!use_graph && !use_multicast);
      const auto& push = comm.get_push_obj();
      const auto two_shot = algo == "2shot_push";
      const auto chunk_vecs = div_ceil(num_vecs, kWorldSize);
      Tensor out = ffi::empty_like(in);
      const auto params = PushParams{
          .input = in.data_ptr(),
          .output = out.data_ptr(),
          .num_vecs = num_vecs,
          .rank = push.rank,
          .ws = push.get_workspace<kWorldSize>(two_shot ? int64_t{2} * chunk_vecs * 16 : nbytes),
      };
      const auto kernel = two_shot ? all_reduce_2shot_push_kernel<T, kWorldSize, kUsePDL>
                                   : all_reduce_1shot_push_kernel<T, kWorldSize, kUsePDL>;
      // the grid is bound to the counter array and must stay constant
      auto block_size = choose_block_size(num_vecs);
#ifdef USE_ROCM
      // MI350X: from 128 KB a full block per CU beats the narrower pick for
      // 1shot_push (TP4/6 -4..-8%, TP8 up to -22%); 2shot_push does not gain
      if (!two_shot && num_vecs >= 8192) block_size = 1024u;
      // MI450 (TP8, 64-block grid): from 512 KB the rule picks 512-1024 threads for
      // 2shot_push; with ranks arriving skewed 256 is up to 40% faster up to 4 MB.
      // Larger sizes keep the rule.
      if (two_shot && is_gfx1250() && nbytes <= int64_t{4} << 20) block_size = 256u;
      if constexpr (kWorldSize <= 8) {
        if (two_shot && is_gfx1250() && nbytes >= kTwoShotPushUnrolledMinBytes<kWorldSize>) {
          LaunchKernel(push.num_blocks, kTwoShotPushThreads, stream)  //
              .enable_pdl(kUsePDL)(all_reduce_2shot_push_unrolled_kernel<T, kWorldSize, kUsePDL>, params);
          return out;
        }
      }
#endif
      if (two_shot) {
        // at least `kWorldSize - 1` warps in all, 1 for each peer
        block_size = std::max(block_size, 32u * div_ceil(kWorldSize - 1, push.num_blocks));
      }
      LaunchKernel(push.num_blocks, block_size, stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
      return out;
    }

    const auto& pull = comm.get_pull_obj();
    const auto graph_params = use_graph ? graph_params_opt.value().data_ptr() : nullptr;
    // only 2shot pull + graph mode forces inplace implementation
    const auto is_inplace = use_graph && algo == "2shot_pull";
    Tensor out = is_inplace ? in : ffi::empty_like(in);
    const auto params = PullParams{
        .output = out.data_ptr(),
        .num_vecs = num_vecs,
        .rank = pull.rank,
        .graph_params = static_cast<void* const*>(graph_params),
        // Graph mode reduces over the caller's own registered buffers and only
        // barriers on this plane; the eager modes stage the input through it.
        .ws = pull.get_workspace<kWorldSize>(use_graph ? 0 : nbytes),
    };
    const auto& ws = params.ws;
    CHECK_HOST(!use_multicast || ws.mc_workspace != nullptr) << "multicast needs a plane with an mc workspace";

    const auto cuda_memcpy = [&](void* dst, const void* src) {
      if constexpr (SGL_ARCH_HOPPER_OR_GREATER) {  // PDL memcpy is faster
        // based on micro benchmark, only enable when batch size is small + aligned
        constexpr int64_t threshold_MB = SGL_ARCH_BLACKWELL_OR_GREATER ? 1024 : 8;
        if (nbytes % device::kMaxVecBytes == 0 && nbytes <= threshold_MB * 1024 * 1024) {
          const auto copy_kernel = memcpy_kernel<device::kMaxVecBytes>;
          const uint32_t num_copy_vecs = nbytes / device::kMaxVecBytes;
          const uint32_t num_copy_threads = 128u;
          const uint32_t num_copy_blocks = div_ceil(num_copy_vecs, num_copy_threads);
          LaunchKernel(num_copy_blocks, num_copy_threads, stream)
              .enable_pdl(kUsePDL)(copy_kernel, dst, src, num_copy_vecs);
          return;
        }
      }
      // safe fallback to cudaMemcpyAsync for large size or older architecture
      CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    };
    const uint32_t num_blocks = comm.get_pull_blocks();
    const auto local_workspace = ws.workspaces[pull.rank];

    using LS = LoadStoreImpl<vec_t, kWorldSize, /*kUseGraph=*/false>;
    using LS_GRAPH = LoadStoreImpl<vec_t, kWorldSize, /*kUseGraph=*/true>;
    using MC = MultiCastImpl<vec_t, kWorldSize, /*kUseGraph=*/false>;
    if (algo == "1shot_pull") {
      // first copy to the workspace
      CHECK_HOST(!use_multicast);
      if (!use_graph) cuda_memcpy(local_workspace, in.data_ptr());
      const auto kernel = use_graph ? all_reduce_1shot_pull_kernel<LS_GRAPH, T, kWorldSize, kUsePDL>
                                    : all_reduce_1shot_pull_kernel<LS, T, kWorldSize, kUsePDL>;
      // then launch kernel to reduce and write to output
      LaunchKernel(num_blocks, choose_block_size(num_vecs), stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
    } else /* algo == "2shot_pull" */ {
      const uint32_t avg_vecs = div_ceil(num_vecs, kWorldSize);
      // first copy to the workspace
      if (!use_graph) cuda_memcpy(local_workspace, in.data_ptr());

      // then launch kernel to reduce in the workspace
      if (use_multicast) {
        CHECK_HOST(!use_graph);
        const auto kernel = all_reduce_2shot_pull_kernel<MC, T, kWorldSize, kUsePDL>;
        // NOTE: too much traffic will degrade performance in multicast impl
        constexpr uint32_t kMulticastNumThreads = 512u;
        const auto num_blocks = comm.get_pull_multicast_blocks();
        LaunchKernel(num_blocks, kMulticastNumThreads, stream)  //
            .enable_pdl(kUsePDL)(kernel, params);
      } else {
        const auto kernel = use_graph ? all_reduce_2shot_pull_kernel<LS_GRAPH, T, kWorldSize, kUsePDL>
                                      : all_reduce_2shot_pull_kernel<LS, T, kWorldSize, kUsePDL>;
        LaunchKernel(num_blocks, choose_block_size(avg_vecs), stream)  //
            .enable_pdl(kUsePDL)(kernel, params);
      }

      // finally copy from the workspace to output
      if (!use_graph) cuda_memcpy(out.data_ptr(), local_workspace);
    }
    return out;
  }
};

}  // namespace sglang
