"""MLX pixel losses for 2D/3D gaussian fitting.

``pixel_loss`` (2D) and ``pixel_loss_3d`` share the same form as 3DGS
training: ``(1 - w) * L1 + w * (1 - SSIM)``. SSIM uses the standard 11x11
sigma=1.5 gaussian window, computed as two separable 1D depthwise
convolutions (121 -> 22 taps per pixel, msplat's formulation) in pure MLX
ops — at 128x128 the convs are negligible next to the rasterizer, so no
custom kernel is needed. At 512p SSIM became the bottleneck, so the public
``ssim`` now routes through a fused Metal custom-function port of
``fused-ssim``. Only the pixel path is implemented; diffusion guidance was
intentionally not ported to MLX (see README).
"""

import math

import mlx.core as mx

from .gaussian import build_L
from .rendering2d_fused import rasterize_fused
from .rendering2dgs import project_gaussians_2dgs  # type: ignore[import-not-found]
from .rendering2dgs_fused import rasterize2dgs_fused  # type: ignore[import-not-found]
from .rendering3d import project_gaussians
from .rendering3d_fused import rasterize3d_fused
from .ssim_fused import ssim_fused  # type: ignore[import-not-found]

# Standard SSIM constants (images in [0, 1]).
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2


def _ssim_windows():
    """(3, 1, 11, 1) horizontal and (3, 11, 1, 1) vertical depthwise conv
    weights for the normalized 1D gaussian window, cached and evaluated once."""
    half = _SSIM_WINDOW // 2
    g = [math.exp(-((x - half) ** 2) / (2 * _SSIM_SIGMA**2)) for x in range(_SSIM_WINDOW)]
    g = mx.array(g, dtype=mx.float32)
    g = g / mx.sum(g)
    wh = mx.broadcast_to(g.reshape(1, 1, _SSIM_WINDOW, 1), (3, 1, _SSIM_WINDOW, 1))
    wv = mx.broadcast_to(g.reshape(1, _SSIM_WINDOW, 1, 1), (3, _SSIM_WINDOW, 1, 1))
    mx.eval(wh, wv)
    return wh, wv


_SSIM_WH, _SSIM_WV = _ssim_windows()


def _gauss_blur(x):
    """Separable 11x11 gaussian blur, 'same' padding (matches the original
    3DGS ``ssim``, which pads by window // 2). Accepts (H, W, 3) or a
    camera batch (B, H, W, 3) — conv2d is batched natively in NHWC."""
    half = _SSIM_WINDOW // 2
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]  # (1, H, W, 3)
    x = mx.conv2d(x, _SSIM_WH, padding=(0, half), groups=3)
    x = mx.conv2d(x, _SSIM_WV, padding=(half, 0), groups=3)
    return x[0] if squeeze else x


def ssim(img1, img2):
    """Mean SSIM between two (H, W, 3) images in [0, 1] (11x11 gaussian
    window, sigma 1.5 — the 3DGS training convention). Batched
    (B, H, W, 3) inputs return the mean over the whole batch."""
    return ssim_fused(img1, img2)


def _blended_loss(rendered, target, ssim_weight):
    """3DGS-style ``(1 - w) * L1 + w * (1 - SSIM)``; pure L1 when w == 0."""
    l1 = mx.mean(mx.abs(rendered - target))
    if ssim_weight > 0:
        return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(rendered, target))
    return l1


def distortion_l1_loss(rendered_distortion):
    """Return the non-negative 2DGS/Mip-NeRF-360 distortion objective.

    The fused prefix-sum expression assumes samples are ordered by their
    per-pixel intersection depth. The compositor, like gsplat, orders surfels
    by center depth; highly tilted surfels can therefore cross at a pixel and
    produce a signed negative residual even though the underlying pairwise L1
    objective is non-negative. Taking the residual magnitude is identical for
    correctly ordered pixels, recovers the exact sign for a crossed pair, and
    retains a corrective gradient without allowing training to optimize an
    unbounded negative surrogate.
    """
    return mx.mean(mx.abs(rendered_distortion))


def _depth_to_normal_camera(depths, K):
    """Surface normals from z-depth in camera coordinates (gsplat 2DGS convention)."""
    squeeze = depths.ndim == 3
    if squeeze:
        depths = depths[None]
        K = K[None] if K.ndim == 2 else K
    batch, height, width = depths.shape[0], depths.shape[1], depths.shape[2]
    if height < 3 or width < 3:
        normals = mx.zeros((batch, height, width, 3), dtype=depths.dtype)
        return normals[0] if squeeze else normals

    x = mx.arange(width, dtype=depths.dtype) + 0.5
    y = mx.arange(height, dtype=depths.dtype) + 0.5
    fx = K[..., 0, 0].reshape(batch, 1, 1)
    fy = K[..., 1, 1].reshape(batch, 1, 1)
    cx = K[..., 0, 2].reshape(batch, 1, 1)
    cy = K[..., 1, 2].reshape(batch, 1, 1)
    xdir = (x.reshape(1, 1, width) - cx) / fx
    ydir = (y.reshape(1, height, 1) - cy) / fy
    xdir = mx.broadcast_to(xdir, (batch, height, width))
    ydir = mx.broadcast_to(ydir, (batch, height, width))
    dirs = mx.stack([xdir, ydir, mx.ones_like(xdir)], axis=-1)
    points = depths * dirs

    dx = points[:, 2:, 1:-1, :] - points[:, :-2, 1:-1, :]
    dy = points[:, 1:-1, 2:, :] - points[:, 1:-1, :-2, :]
    nx = dx[..., 1] * dy[..., 2] - dx[..., 2] * dy[..., 1]
    ny = dx[..., 2] * dy[..., 0] - dx[..., 0] * dy[..., 2]
    nz = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    normals = mx.stack([nx, ny, nz], axis=-1)
    normal_len = mx.sqrt(mx.sum(normals * normals, axis=-1, keepdims=True) + 1e-20)
    normals = normals / normal_len

    zeros_h = mx.zeros((batch, 1, width - 2, 3), dtype=depths.dtype)
    normals = mx.concatenate([zeros_h, normals, zeros_h], axis=1)
    zeros_w = mx.zeros((batch, height, 1, 3), dtype=depths.dtype)
    normals = mx.concatenate([zeros_w, normals, zeros_w], axis=2)
    return normals[0] if squeeze else normals


def pixel_loss(
    means,
    log_diag,
    offdiag,
    colors,
    background_color,
    target_image,
    ssim_weight=0.2,
):
    """L1 loss between the rasterized Gaussians and a target image.

    Args:
        means: (N, 2) gaussian centers in pixel coords.
        log_diag: (N, 2) log of L's diagonal. ``L[i,i] = exp(log_diag[i])``.
        offdiag: (N,) L's off-diagonal entry ``L[i, 1, 0]``.
        colors: (N, 3) RGB values per gaussian.
        background_color: (1, 1, 3) background RGB.
        target_image: (H, W, 3) target RGB in [0, 1].
        ssim_weight: SSIM blend weight: ``(1 - w) * L1 + w * (1 - SSIM)``.

    Returns:
        (loss, rendered_gaussians): the blended loss and the (H, W, 3)
        rasterized image.
    """
    L = build_L(log_diag, offdiag)
    covariances = L @ mx.transpose(L, (0, 2, 1))

    height, width, _ = target_image.shape
    # Broadcast (1, 1, 3) -> (H, W, 3) without materializing the full array.
    background = mx.broadcast_to(background_color, (height, width, 3))

    # Fused Metal-kernel rasterizer (gsplat-style); same math as
    # rendering2d.rasterize, which stays as the dense reference
    # implementation (see EXPERIMENTS.md for the numerics comparison).
    rendered_gaussians, _, _ = rasterize_fused(means, covariances, colors, background, height, width)
    loss = _blended_loss(rendered_gaussians, target_image, ssim_weight)
    return loss, rendered_gaussians


def pixel_loss_3d(
    means3d,
    log_scales,
    quats,
    opacities_raw,
    colors,
    target_image,
    viewmat,
    K,
    ssim_weight=0.1,
    means2d_offset=None,
    means2d_absgrad_sink=None,
    bin_pad=None,
    bin_capacity=None,
    return_counts=False,
):
    """L1 loss between alpha-composited 3D Gaussians and a target image.

    The 3D analog of :func:`pixel_loss`, following gsplat's
    ``examples/image_fitting.py``: gaussians live in world space, are
    EWA-projected through a fixed pinhole camera and alpha-composited in
    depth order (fused Metal kernels, see rendering3d_fused.py). Opacities
    and colors are optimized in logit space (sigmoid applied here, like the
    gsplat example); the background is fixed black.

    Args:
        means3d: (N, 3) world-space centers.
        log_scales: (N, 3) log of per-axis scales.
        quats: (N, 4) wxyz quaternions (normalized inside the projection).
        opacities_raw: (N,) opacity logits.
        colors: (N, 3) or (B,N,3) activated per-view RGB from spherical harmonics.
        target_image: (H, W, 3) target RGB in [0, 1], or a camera batch
            (B, H, W, 3) paired with batched ``viewmat``/``K`` — the whole
            batch renders in one kernel launch (gsplat's [..., C, N]
            convention) and the loss is the mean over all views.
        viewmat: (4, 4) world-to-camera matrix, or (B, 4, 4).
        K: (3, 3) camera intrinsics, or (B, 3, 3).
        ssim_weight: SSIM blend weight, as in :func:`pixel_loss`.
        means2d_offset: optional (N, 2) zeros added to the projected means.
            Its gradient equals the net screen-space means2d gradient — the
            MLX equivalent of gsplat's ``retain_grad`` on means2d. With a
            camera batch it broadcasts over views and its gradient sums
            over them (same for ``means2d_absgrad_sink``).
        means2d_absgrad_sink: optional ignored (N, 2) zero tensor. Its custom
            VJP gradient accumulates per-pixel absolute means2d-gradient
            contributions from the fused rasterizer, matching gsplat's
            ``absgrad`` densification signal.
        bin_pad: optional per-Gaussian compact capacity multiplier kept for
            compatibility; ``None`` with ``bin_capacity=None`` is the old exact
            all-tiles path.
        bin_capacity: optional static compact-bin capacity (number of sorted
            intersection keys, including INVALID tail).
        return_counts: append exact per-view/per-Gaussian tile-intersection
            counts in parameter order. Compact rendering reuses builder counts.

    Returns:
        ``(loss, rendered)`` by default, or ``(loss, rendered, counts)`` when
        ``return_counts=True``.
    """
    height, width = target_image.shape[-3], target_image.shape[-2]
    means2d, conics, depths = project_gaussians(means3d, log_scales, quats, viewmat, K, width, height)
    if means2d_offset is not None:
        means2d = means2d + means2d_offset
    rendered_result = rasterize3d_fused(
        means2d,
        conics,
        mx.sigmoid(opacities_raw),
        colors,
        mx.zeros((3,), dtype=mx.float32),
        depths,
        height,
        width,
        absgrad_sink=means2d_absgrad_sink,
        bin_pad=bin_pad,
        bin_capacity=bin_capacity,
        return_counts=return_counts,
    )
    if return_counts:
        rendered, counts = rendered_result
    else:
        rendered, counts = rendered_result, None
    loss = _blended_loss(rendered, target_image, ssim_weight)
    return (loss, rendered, counts) if return_counts else (loss, rendered)


def pixel_loss_2dgs(
    means3d,
    log_scales,
    quats,
    opacities_raw,
    colors,
    target_image,
    viewmat,
    K,
    ssim_weight=0.1,
    means2d_offset=None,
    means2d_absgrad_sink=None,
    bin_pad=None,
    bin_capacity=None,
    normal_weight=0.0,
    distortion_weight=0.0,
    normal_depth_mode="expected",
    return_counts=False,
    return_components=False,
):
    """2DGS surfel photometric loss plus optional geometry regularizers.

    ``normal_weight`` enables gsplat-style normal consistency between rendered
    surfel normals and normals estimated from rendered depth. ``distortion_weight``
    adds the 2DGS/Mip-NeRF-360 distortion regularizer from the fused rasterizer.
    Both default to zero, preserving the old RGB-only loss.
    """
    height, width = target_image.shape[-3], target_image.shape[-2]
    radii, means2d, depths, ray_transforms, normals = project_gaussians_2dgs(
        means3d, log_scales, quats, viewmat, K, width, height
    )
    if means2d_offset is not None:
        means2d = means2d + means2d_offset
    need_aux = normal_weight > 0.0 or distortion_weight > 0.0
    rendered_or_pair = rasterize2dgs_fused(
        means2d,
        ray_transforms,
        mx.sigmoid(opacities_raw),
        colors,
        mx.zeros((3,), dtype=mx.float32),
        depths,
        radii,
        height,
        width,
        absgrad_sink=means2d_absgrad_sink,
        bin_pad=bin_pad,
        bin_capacity=bin_capacity,
        normals=normals,
        return_aux=need_aux,
        return_counts=return_counts,
    )
    if need_aux:
        if return_counts:
            rendered, aux, counts = rendered_or_pair
        else:
            rendered, aux = rendered_or_pair
            counts = None
    else:
        if return_counts:
            rendered, counts = rendered_or_pair
        else:
            rendered, counts = rendered_or_pair, None
        aux = None
    photometric_loss = _blended_loss(rendered, target_image, ssim_weight)
    loss = photometric_loss
    normal_loss = mx.zeros((), dtype=loss.dtype)
    distortion_loss = mx.zeros((), dtype=loss.dtype)
    if aux is not None and normal_weight > 0.0:
        if normal_depth_mode == "median":
            depth_for_normal = aux["median_depth"]
        elif normal_depth_mode == "expected":
            depth_for_normal = aux["depth"]
        else:
            raise ValueError("normal_depth_mode must be 'expected' or 'median'")
        surface_normals = _depth_to_normal_camera(depth_for_normal, K)
        surface_normals = surface_normals * mx.stop_gradient(aux["alpha"])
        normal_error = 1.0 - mx.sum(aux["normals"] * surface_normals, axis=-1)
        normal_loss = mx.mean(normal_error)
        loss = loss + normal_weight * normal_loss
    if aux is not None and distortion_weight > 0.0:
        distortion_loss = distortion_l1_loss(aux["distortion"])
        loss = loss + distortion_weight * distortion_loss
    components = {
        "photometric": photometric_loss,
        "normal_consistency": normal_loss,
        "distortion": distortion_loss,
    }
    if return_counts and return_components:
        return loss, rendered, counts, components
    if return_counts:
        return loss, rendered, counts
    if return_components:
        return loss, rendered, components
    return loss, rendered
