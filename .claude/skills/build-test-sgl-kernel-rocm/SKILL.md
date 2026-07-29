---
name: build-test-sgl-kernel-rocm
description: Verify a lmsysorg ROCm docker image targets the right AMD GPU arch (gfx942/gfx950), then build sgl-kernel from source and run its tests inside the container on AMD Instinct GPUs. Use when asked to check gfx support of an sglang-rocm image, build sgl-kernel for a gfx target, or run/port sgl-kernel tests (especially allreduce) on ROCm/MI300/MI350.
---

# Build & Test sgl-kernel on AMD ROCm

Workflow for validating an `lmsysorg/sglang-rocm` docker image and building/testing
`sgl-kernel` from a mounted source tree on AMD Instinct GPUs.

## Key facts

- AMD gfx targets: `gfx942` = MI300X/MI325 (CDNA3), `gfx950` = MI350/MI355 (CDNA4), `gfx1250` = MI400 series. Image tags encode this: `mi30x`→gfx942, `mi35x`→gfx950, `mi45x`→gfx1250.
- `sgl-kernel` builds on ROCm via `setup_rocm.py` (NOT the default `setup.py`). It only accepts `gfx942` or `gfx950`.
- Host must expose `/dev/kfd` and `/dev/dri`; check GPUs with `rocm-smi --showproductname`.

## Step 1: Verify the image targets the intended gfx arch

Env vars alone can lie; scan the actually-compiled code objects. `torch/lib` is a broad
prebuilt wheel (contains many archs), so the decisive check is `sgl_kernel` and `aiter`.

```bash
docker run --rm --entrypoint bash <image> -c '
  env | grep -iE "PYTORCH_ROCM_ARCH|GPU_ARCH"
  SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
  echo "== sgl_kernel =="; grep -rhoaE "gfx(9[0-9]{2}|1[0-9]{3})" $SP/sgl_kernel | sort | uniq -c
  A=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))" 2>/dev/null)
  echo "== aiter =="; grep -rhoaE "gfx(9[0-9]{2}|1[0-9]{3})" "$A" | sort | uniq -c'
```

An image genuinely supports a target only if `sgl_kernel` (and `aiter`, if used) contain
that gfx string. An `aiter` import that fails on `rocminfo` when run WITHOUT a GPU is
expected — attach GPUs (Step 2) or just grep the files.

## Step 2: Launch a container with GPUs + repo mounted

```bash
docker run -d --name sgl-dev \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host --shm-size 16G \
  -v <repo_abs_path>:/repo --entrypoint sleep <image> infinity
docker exec sgl-dev python3 -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).gcnArchName)'
```

## Step 3: Build sgl-kernel for the gfx target

Build from a container-local copy so the mounted host repo stays free of build artifacts.
`torch.cuda` auto-detects the arch when GPUs are attached; `AMDGPU_TARGET` is the fallback.

```bash
docker exec sgl-dev bash -c '
  rm -rf /work && cp -r /repo/sgl-kernel /work && cd /work/sgl-kernel
  AMDGPU_TARGET=gfx950 MAX_JOBS=$(nproc) python3 setup_rocm.py build_ext --inplace'
```

Success prints `building '\''sgl_kernel.common_ops'\''` and links `common_ops.*.so` into
`python/sgl_kernel/`. The `--amdgpu-target` warning and `[-Wpass-failed]` warnings are benign.

## Step 4: Run tests, pointing PYTHONPATH at the fresh build

```bash
docker exec sgl-dev bash -c '
  export PYTHONPATH=/work/sgl-kernel/python:/repo/python
  cd /repo && python3 -m pytest -s -q sgl-kernel/tests/test_<name>.py'
```

## ROCm vs CUDA gotcha: custom all-reduce

`sgl-kernel/tests/test_custom_allreduce.py` and CUDA code use a `libcudart` IPC wrapper
(`CudaRTLibrary`) that does not exist on ROCm (only `libamdhip64` is loaded), and the
custom-AR op ABI differs by platform:

| | CUDA | ROCm |
|---|---|---|
| init | `init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)` | `init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink)` |
| buffers | raw IPC pointers (libcudart) | `allocate_meta_buffer` + IPC handle/offset exchange |
| reduce | unified `all_reduce(...)` | `all_reduce_reg` / `all_reduce_unreg` (or `deterministic_*`) |

Do NOT try to make the raw-ops CUDA path run on ROCm. Instead branch on
`torch.version.hip is not None` and drive the kernel through the production
`sglang.srt.distributed.device_communicators.custom_all_reduce.CustomAllreduce`
class (set `envs.SGLANG_USE_1STAGE_ALLREDUCE.set("1")` to force the deterministic
1-stage path so the kernel is always selected regardless of xGMI/NVLink detection),
comparing against the RCCL `dist.all_reduce` reference. `CustomAllreduce` needs a
gloo sub-group (`new_group(backend="gloo")`) for its IPC metadata exchange.

Existing ROCm-native allreduce coverage lives in
`test/registered/jit/test_amd_deterministic_custom_allreduce.py` (determinism) and
`sgl-kernel/benchmark/bench_amd_deterministic_allreduce.py` (latency).

### Benchmark gotcha

In `bench_amd_deterministic_allreduce.py`, if a "Custom AR" latency looks identical to
the "RS+AG" row (~4-9x slower than the real kernel), the custom path is falling back to
NCCL/RCCL reduce-scatter + all-gather instead of running the kernel — confirm
`reduce_scatter_then_all_gather` actually uses its `custom_ar` argument (call
`custom_ar.custom_all_reduce(tensor)` when provided). The genuine kernel latency is the
"Deterministic Kernel" column. Also note the large-batch cliff (e.g. batch 512 / 16 MB
input hitting `max_size`) is real: it re-registers the buffer each call.

## Cleanup

```bash
docker rm -f sgl-dev
```
