# Native MLX/Metal evaluation follow-up

TorchMetrics/torchvision and Open3D have been removed from the runtime path.

- [x] Reimplement clamped PSNR, SSIM, and LPIPS-Alex inference in MLX. The
  AlexNet conversion matches TorchMetrics to 1.5e-8 on the stored fixture.
- [x] Replace Open3D scalable TSDF integration and mesh extraction with sparse
  MLX-streamed TSDF fusion and deterministic marching tetrahedra. MLX indexed
  reduction uses ``array.at[idx].add``; sparse key compaction/block ownership
  remains host-orchestrated because MLX 0.31 has no dynamic ``unique`` or hash
  table container.
  The open-addressed Metal int3 hash set is 13.3x faster than ``np.unique``
  on a 1M-row / 100K-unique activation fixture, with identical output.
- [ ] Reimplement deterministic mesh sampling, nearest-neighbour DTU scoring,
  observation-volume filtering, plane clipping, and projection RQ without
  SciPy/host numerical kernels.
- Keep SciPy and official MATLAB results only as fixture generators and parity
  oracles, then move them out of the runtime dependency path.
- Benchmark stream execution, unified-memory peak use, compilation time, and
  end-to-end evaluation time on Apple Silicon before promoting the native path.
- Preserve the current JSON schemas and deterministic seeds so historical
  runs remain directly comparable.
