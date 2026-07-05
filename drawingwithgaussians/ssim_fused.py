"""Fused Metal SSIM for MLX NHWC images.

This ports the 2D Metal structure from
``/Users/glebsterkin/repos/fused-ssim`` to ``mx.fast.metal_kernel`` and the
repo's NHWC image layout. The public function returns the mean SSIM matching
``losses.ssim``'s 11x11 sigma=1.5 zero-padded reference, while the custom VJP
returns gradients only for ``img1`` (the rendered image); training targets are
constants in this project.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_BLOCK_X = 16
_BLOCK_Y = 16
_CORE_CACHE: dict[tuple[int, int, int, int], Any] = {}

_HEADER = """
constant uint BLOCK_X = 16;
constant uint BLOCK_Y = 16;
constant uint HALO = 5;
constant uint SHARED_X = BLOCK_X + 2 * HALO;
constant uint SHARED_Y = BLOCK_Y + 2 * HALO;
constant uint CONV_X = BLOCK_X;
constant uint CONV_Y = SHARED_Y;
constant float C1 = 0.0001f;
constant float C2 = 0.0009f;

constant float cGauss[11] = {
    0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f,
    0.21300553f, 0.26601172f, 0.21300553f, 0.10936069f,
    0.03600077f, 0.00759876f, 0.00102838f
};

inline uint idx_nhwc(uint b, uint y, uint x, uint c, uint H, uint W, uint C) {
    return ((b * H + y) * W + x) * C + c;
}

inline float get_pix_value(
    device const float* img,
    uint b,
    uint c,
    int y,
    int x,
    uint B,
    uint H,
    uint W,
    uint C
) {
    if (b >= B || x < 0 || x >= (int)W || y < 0 || y >= (int)H) return 0.0f;
    return img[idx_nhwc(b, (uint)y, (uint)x, c, H, W, C)];
}
"""

_FORWARD_SRC = """
    uint lid = thread_index_in_threadgroup;
    uint3 tg = threadgroup_position_in_grid;
    uint tx = lid % BLOCK_X;
    uint ty = lid / BLOCK_X;
    uint B = (uint)sizes[0];
    uint H = (uint)sizes[1];
    uint W = (uint)sizes[2];
    uint C = (uint)sizes[3];
    uint px = tg.x * BLOCK_X + tx;
    uint py = tg.y * BLOCK_Y + ty;
    uint b = tg.z;
    uint pix_id = py * W + px;

    threadgroup float sTile[SHARED_Y][SHARED_X][2];
    threadgroup float xconv[CONV_Y][CONV_X][5];

    for (uint c = 0; c < C; ++c) {
        for (uint tid = lid; tid < SHARED_X * SHARED_Y; tid += BLOCK_X * BLOCK_Y) {
            uint local_y = tid / SHARED_X;
            uint local_x = tid % SHARED_X;
            int gy = (int)(tg.y * BLOCK_Y + local_y) - (int)HALO;
            int gx = (int)(tg.x * BLOCK_X + local_x) - (int)HALO;
            float X = get_pix_value(img1, b, c, gy, gx, B, H, W, C);
            float Y = get_pix_value(img2, b, c, gy, gx, B, H, W, C);
            sTile[local_y][local_x][0] = X;
            sTile[local_y][local_x][1] = Y;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        uint lx = tx + HALO;
        for (uint pass = 0; pass < 2; ++pass) {
            uint yy = ty + pass * BLOCK_Y;
            if (yy < CONV_Y) {
                float sumX = 0.0f, sumX2 = 0.0f, sumY = 0.0f, sumY2 = 0.0f, sumXY = 0.0f;
                for (uint d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float xl = sTile[yy][lx - d][0];
                    float xr = sTile[yy][lx + d][0];
                    float yl = sTile[yy][lx - d][1];
                    float yr = sTile[yy][lx + d][1];
                    sumX += (xl + xr) * w;
                    sumX2 += (xl * xl + xr * xr) * w;
                    sumY += (yl + yr) * w;
                    sumY2 += (yl * yl + yr * yr) * w;
                    sumXY += (xl * yl + xr * yr) * w;
                }
                float cx = sTile[yy][lx][0];
                float cy = sTile[yy][lx][1];
                float wc = cGauss[HALO];
                sumX += cx * wc;
                sumX2 += cx * cx * wc;
                sumY += cy * wc;
                sumY2 += cy * cy * wc;
                sumXY += cx * cy * wc;
                xconv[yy][tx][0] = sumX;
                xconv[yy][tx][1] = sumX2;
                xconv[yy][tx][2] = sumY;
                xconv[yy][tx][3] = sumY2;
                xconv[yy][tx][4] = sumXY;
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if (px < W && py < H && b < B) {
            uint ly = ty + HALO;
            float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f, out4 = 0.0f;
            for (uint d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                threadgroup float* top = xconv[ly - d][tx];
                threadgroup float* bot = xconv[ly + d][tx];
                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            float wc = cGauss[HALO];
            threadgroup float* ctr = xconv[ly][tx];
            out0 += ctr[0] * wc;
            out1 += ctr[1] * wc;
            out2 += ctr[2] * wc;
            out3 += ctr[3] * wc;
            out4 += ctr[4] * wc;

            float mu1 = out0;
            float mu2 = out2;
            float mu1_sq = mu1 * mu1;
            float mu2_sq = mu2 * mu2;
            float sigma1_sq = out1 - mu1_sq;
            float sigma2_sq = out3 - mu2_sq;
            float sigma12 = out4 - mu1 * mu2;
            float A = mu1_sq + mu2_sq + C1;
            float Bv = sigma1_sq + sigma2_sq + C2;
            float Cv = 2.0f * mu1 * mu2 + C1;
            float Dv = 2.0f * sigma12 + C2;
            float val = (Cv * Dv) / (A * Bv);

            uint out_idx = ((b * H * W + pix_id) * C + c);
            ssim_map[out_idx] = val;
            dm_dmu1[out_idx] = (2.0f * mu2 * Dv) / (A * Bv)
                - (2.0f * mu2 * Cv) / (A * Bv)
                - (2.0f * mu1 * Cv * Dv) / (A * A * Bv)
                + (2.0f * mu1 * Cv * Dv) / (A * Bv * Bv);
            dm_dsigma1_sq[out_idx] = (-Cv * Dv) / (A * Bv * Bv);
            dm_dsigma12[out_idx] = (2.0f * Cv) / (A * Bv);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
"""

_BACKWARD_SRC = """
    uint lid = thread_index_in_threadgroup;
    uint3 tg = threadgroup_position_in_grid;
    uint tx = lid % BLOCK_X;
    uint ty = lid / BLOCK_X;
    uint B = (uint)sizes[0];
    uint H = (uint)sizes[1];
    uint W = (uint)sizes[2];
    uint C = (uint)sizes[3];
    uint px = tg.x * BLOCK_X + tx;
    uint py = tg.y * BLOCK_Y + ty;
    uint b = tg.z;
    uint pix_id = py * W + px;

    threadgroup float sData[3][SHARED_Y][SHARED_X];
    threadgroup float sScratch[CONV_Y][CONV_X][3];

    for (uint c = 0; c < C; ++c) {
        float p1 = 0.0f, p2 = 0.0f;
        if (px < W && py < H && b < B) {
            p1 = get_pix_value(img1, b, c, (int)py, (int)px, B, H, W, C);
            p2 = get_pix_value(img2, b, c, (int)py, (int)px, B, H, W, C);
        }

        for (uint tid = lid; tid < SHARED_X * SHARED_Y; tid += BLOCK_X * BLOCK_Y) {
            uint row = tid / SHARED_X;
            uint col = tid % SHARED_X;
            int gy = (int)(tg.y * BLOCK_Y + row) - (int)HALO;
            int gx = (int)(tg.x * BLOCK_X + col) - (int)HALO;
            float chain = get_pix_value(dmap, b, c, gy, gx, B, H, W, C);
            float vmu = get_pix_value(dm_dmu1, b, c, gy, gx, B, H, W, C);
            float vs1 = get_pix_value(dm_dsigma1_sq, b, c, gy, gx, B, H, W, C);
            float vs12 = get_pix_value(dm_dsigma12, b, c, gy, gx, B, H, W, C);
            sData[0][row][col] = vmu * chain;
            sData[1][row][col] = vs1 * chain;
            sData[2][row][col] = vs12 * chain;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        uint lx = tx + HALO;
        for (uint pass = 0; pass < 2; ++pass) {
            uint yy = ty + pass * BLOCK_Y;
            if (yy < CONV_Y) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
                for (uint d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    acc0 += (sData[0][yy][lx - d] + sData[0][yy][lx + d]) * w;
                    acc1 += (sData[1][yy][lx - d] + sData[1][yy][lx + d]) * w;
                    acc2 += (sData[2][yy][lx - d] + sData[2][yy][lx + d]) * w;
                }
                float wc = cGauss[HALO];
                acc0 += sData[0][yy][lx] * wc;
                acc1 += sData[1][yy][lx] * wc;
                acc2 += sData[2][yy][lx] * wc;
                sScratch[yy][tx][0] = acc0;
                sScratch[yy][tx][1] = acc1;
                sScratch[yy][tx][2] = acc2;
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if (px < W && py < H && b < B) {
            uint ly = ty + HALO;
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
            for (uint d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                threadgroup float* top = sScratch[ly - d][tx];
                threadgroup float* bot = sScratch[ly + d][tx];
                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            float wc = cGauss[HALO];
            threadgroup float* ctr = sScratch[ly][tx];
            sum0 += ctr[0] * wc;
            sum1 += ctr[1] * wc;
            sum2 += ctr[2] * wc;
            dimg1[(b * H * W + pix_id) * C + c] = sum0 + (2.0f * p1) * sum1 + p2 * sum2;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
"""

_k_fwd = mx.fast.metal_kernel(
    name="fused_ssim_forward_nhwc",
    input_names=["img1", "img2", "sizes"],
    output_names=["ssim_map", "dm_dmu1", "dm_dsigma1_sq", "dm_dsigma12"],
    header=_HEADER,
    source=_FORWARD_SRC,
)

_k_bwd = mx.fast.metal_kernel(
    name="fused_ssim_backward_nhwc",
    input_names=[
        "img1",
        "img2",
        "dmap",
        "dm_dmu1",
        "dm_dsigma1_sq",
        "dm_dsigma12",
        "sizes",
    ],
    output_names=["dimg1"],
    header=_HEADER,
    source=_BACKWARD_SRC,
)


def _pad(x: int, block: int) -> int:
    return ((x + block - 1) // block) * block


def _core(batch: int, height: int, width: int, channels: int) -> Any:
    key = (batch, height, width, channels)
    cached = _CORE_CACHE.get(key)
    if cached is not None:
        return cached

    grid = (_pad(width, _BLOCK_X), _pad(height, _BLOCK_Y), batch)
    threadgroup = (_BLOCK_X, _BLOCK_Y, 1)
    out_shape = (batch, height, width, channels)

    @mx.custom_function
    def core(img1, img2):
        sizes = mx.array([batch, height, width, channels], dtype=mx.int32)
        return _k_fwd(  # type: ignore[operator]
            inputs=[img1, img2, sizes],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[out_shape] * 4,
            output_dtypes=[mx.float32] * 4,
        )

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        img1, img2 = primals
        dmap = cotangents[0]
        _ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = outputs
        sizes = mx.array([batch, height, width, channels], dtype=mx.int32)
        dimg1 = _k_bwd(  # type: ignore[operator]
            inputs=[img1, img2, dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, sizes],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]
        return dimg1, mx.zeros_like(img2)

    _CORE_CACHE[key] = core
    return core


def ssim_map_fused(img1, img2):
    """Return the per-pixel/channel SSIM map for HWC or BHWC images."""
    if img1.shape != img2.shape:
        raise ValueError(
            f"SSIM inputs must have matching shapes, got {img1.shape} and {img2.shape}"
        )
    squeeze = img1.ndim == 3
    if squeeze:
        img1 = img1[None]
        img2 = img2[None]
    if img1.ndim != 4:
        raise ValueError(f"SSIM expects HWC or BHWC images, got shape {img1.shape}")
    batch, height, width, channels = img1.shape
    ssim_map, _dm_dmu1, _dm_dsigma1_sq, _dm_dsigma12 = _core(
        batch, height, width, channels
    )(img1, img2)
    return ssim_map[0] if squeeze else ssim_map


def ssim_fused(img1, img2):
    """Mean fused SSIM for HWC or BHWC images."""
    return mx.mean(ssim_map_fused(img1, img2))
