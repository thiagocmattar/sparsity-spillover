"""Small environment checks: CUDA, BF16, Triton, C++ extension and graph replay."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


@triton.jit
def threshold_kernel(x, y, n: tl.constexpr, cutoff: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(x + index, index < n, other=0)
    tl.store(y + index, tl.where(value < cutoff, 0, value), index < n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {'python': '3.12', 'torch': '2.11.0+cu128', 'cuda': '12.8',
                'triton': '3.6.0', 'transformers': '5.12.1', 'numpy': '2.5.0'}
    versions = {'python': '.'.join(platform.python_version_tuple()[:2]),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'triton': triton.__version__, 'transformers': transformers.__version__,
                'numpy': np.__version__}
    assert versions == expected, versions
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = []
    with torch.inference_mode():
        for k in (512, 2048):
            x = torch.randn(2048, k, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(k, 512, device='cuda', dtype=torch.bfloat16)
            actual = x @ w
            reference = x.float() @ w.float()
            relative_l2 = float(torch.linalg.vector_norm(actual.float() - reference)
                                / torch.linalg.vector_norm(reference))
            assert relative_l2 < .01
            results.append({'shape': [2048, k, 512], 'relative_l2': relative_l2})
        x = torch.linspace(-1, 1, 4096, device='cuda', dtype=torch.bfloat16)
        cutoff = float(torch.tensor(.1, dtype=torch.bfloat16))
        y = torch.empty_like(x)
        launch = lambda: threshold_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), cutoff, 256)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for offset in (0., .5):
            x.add_(offset)
            graph.replay()
            torch.testing.assert_close(y, torch.where(x < cutoff, 0, x), rtol=0, atol=0)
    extension = load_inline(
        name='sparsity_local_environment_smoke',
        cpp_sources='torch::Tensor increment_cuda(torch::Tensor x);',
        cuda_sources='''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
__global__ void increment_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.f;
}
torch::Tensor increment_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    increment_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
''', functions=['increment_cuda'], extra_cuda_cflags=['-O3'], verbose=True)
    value = torch.arange(1024, device='cuda', dtype=torch.float32)
    torch.testing.assert_close(extension.increment_cuda(value), value + 1, rtol=0, atol=0)
    torch.cuda.synchronize()
    report = {'status': 'passed', 'versions': versions, 'gpu': torch.cuda.get_device_name(),
              'capability': torch.cuda.get_device_capability(), 'bf16_matmuls': results,
              'triton_changed_input_graph_replay': True, 'cuda_extension': True,
              'free_total_bytes': torch.cuda.mem_get_info(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'scope': 'Infrastructure only; no model benchmark or kernel qualification.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
