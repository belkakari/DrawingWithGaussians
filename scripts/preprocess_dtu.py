#!/usr/bin/env python3
"""Create a train-view-only fixed-pose COLMAP reconstruction for DTU SampleSet."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from drawingwithgaussians.dtu import DTUSampleSet, project_points

DEFAULT_DTU_ROOT = Path(__file__).resolve().parents[1] / "inputs" / "dtu"


def _options(obj, values: dict):
    if values:
        obj.mergedict(values)
    return obj


def preprocess(
    dataset: DTUSampleSet,
    cache_root: Path,
    feature_settings: dict,
    matcher_settings: dict,
    max_reprojection_error: float,
    min_track_length: int,
) -> Path:
    import pycolmap

    filter_settings = {
        "max_reprojection_error_native_px": float(max_reprojection_error),
        "min_track_length": int(min_track_length),
    }
    key = dataset.cache_key(feature_settings, matcher_settings, filter_settings)
    root = cache_root / f"scan{dataset.scan}_{key}"
    sparse = root / "sparse" / "0"
    done = root / "manifest.json"
    if done.is_file() and sparse.is_dir():
        return root
    if root.exists():
        shutil.rmtree(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    def stage_image(view: int) -> str:
        source = dataset.image_path(view)
        name = source.name
        destination = images_dir / name
        if destination.exists():
            return name
        if dataset.image_scale == 1.0:
            destination.symlink_to(source)
        else:
            from PIL import Image

            with Image.open(source) as image:
                size = tuple(max(1, round(v * dataset.image_scale)) for v in image.size)
                image.resize(size, Image.Resampling.LANCZOS).save(destination)
        return name

    names = []
    for view in dataset.train_views:
        names.append(stage_image(view))

    database_path = root / "database.db"
    reader = pycolmap.ImageReaderOptions()
    extraction = _options(pycolmap.FeatureExtractionOptions(), feature_settings)
    pycolmap.extract_features(
        database_path,
        images_dir,
        image_names=names,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        reader_options=reader,
        extraction_options=extraction,
    )
    pycolmap.match_exhaustive(
        database_path,
        matching_options=_options(pycolmap.FeatureMatchingOptions(), matcher_settings),
    )

    database = pycolmap.Database.open(database_path)
    db_images = {im.name: im for im in database.read_all_images()}
    reconstruction = pycolmap.Reconstruction()
    for view, name in zip(dataset.train_views, names, strict=True):
        db_image = db_images[name]
        projection = dataset.projection(view)
        from PIL import Image

        with Image.open(dataset.image_path(view)) as image:
            width, height = image.size
        width = round(width * dataset.image_scale)
        height = round(height * dataset.image_scale)
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[projection.K[0, 0], projection.K[1, 1], projection.K[0, 2], projection.K[1, 2]],
            camera_id=int(db_image.camera_id),
        )
        database.update_camera(camera)
        keypoints = np.asarray(database.read_keypoints(int(db_image.image_id)))[:, :2]
        image = pycolmap.Image(
            name=name,
            keypoints=keypoints,
            camera_id=int(db_image.camera_id),
            image_id=int(db_image.image_id),
        )
        reconstruction.add_camera_with_trivial_rig(camera)
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(projection.w2c[:3, :4]))
    database.close()

    sparse.mkdir(parents=True)
    result = pycolmap.triangulate_points(
        reconstruction,
        database_path,
        images_dir,
        sparse,
        clear_points=True,
        refine_intrinsics=False,
    )
    name_to_view = dict(zip(names, dataset.train_views, strict=True))
    removed = 0
    reprojection_errors = []
    for point_id in list(result.point3D_ids()):
        point = result.point3D(point_id)
        valid = (
            point.track.length() >= min_track_length
            and point.has_error()
            and point.error <= max_reprojection_error * dataset.image_scale
        )
        track_errors = []
        for obs in point.track.elements:
            image = result.images[obs.image_id]
            view = name_to_view[image.name]
            P = np.loadtxt(dataset.projection_path(view))
            xy, depth = project_points(P, point.xyz[None])
            valid &= bool(depth[0] > 0 and np.isfinite(xy).all())
            observed = np.asarray(image.point2D(obs.point2D_idx).xy)
            error_native = float(np.linalg.norm(xy[0] - observed / dataset.image_scale))
            track_errors.append(error_native)
            valid &= error_native <= max_reprojection_error
        if not valid:
            result.delete_point3D(point_id)
            removed += 1
        else:
            reprojection_errors.extend(track_errors)

    # Add the seven held-out cameras only after matching/triangulation. They
    # have fixed poses and image files, but no keypoints, descriptors, matches,
    # or point tracks, so RGB evaluation cannot leak into initialization.
    next_camera_id = max((int(v) for v in result.cameras.keys()), default=0) + 1
    next_image_id = max((int(v) for v in result.images.keys()), default=0) + 1
    for offset, view in enumerate(dataset.validation_views):
        name = stage_image(view)
        projection = dataset.projection(view)
        from PIL import Image

        with Image.open(images_dir / name) as image_file:
            width, height = image_file.size
        camera_id = next_camera_id + offset
        image_id = next_image_id + offset
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[projection.K[0, 0], projection.K[1, 1], projection.K[0, 2], projection.K[1, 2]],
            camera_id=camera_id,
        )
        image = pycolmap.Image(name=name, camera_id=camera_id, image_id=image_id)
        result.add_camera_with_trivial_rig(camera)
        result.add_image_with_trivial_frame(image, pycolmap.Rigid3d(projection.w2c[:3, :4]))
    result.write(sparse)
    manifest = {
        "cache_key": key,
        "scan": dataset.scan,
        "lighting": dataset.lighting,
        "image_scale": dataset.image_scale,
        "train_views": dataset.train_views,
        "validation_views": dataset.validation_views,
        "feature_settings": feature_settings,
        "matcher_settings": matcher_settings,
        "filter_settings": filter_settings,
        "num_points": result.num_points3D(),
        "registered_images": result.num_reg_images(),
        "held_out_images_without_features": len(dataset.validation_views),
        "removed_points": removed,
        "original_projection_reprojection_mean_px": float(np.mean(reprojection_errors)),
        "original_projection_reprojection_max_px": float(np.max(reprojection_errors)),
    }
    done.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_DTU_ROOT)
    parser.add_argument("--scan", type=int, choices=(1, 6), required=True)
    parser.add_argument("--lighting", default="3")
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--cache-root", type=Path, default=Path(DEFAULT_DTU_ROOT) / "cache")
    parser.add_argument("--feature-settings", type=json.loads, default={})
    parser.add_argument("--matcher-settings", type=json.loads, default={})
    parser.add_argument("--max-reprojection-error", type=float, default=4.0)
    parser.add_argument("--min-track-length", type=int, default=3)
    args = parser.parse_args()
    dataset = DTUSampleSet(args.root, args.scan, args.lighting, args.image_scale)
    dataset.validate_layout()
    print(
        preprocess(
            dataset,
            args.cache_root,
            args.feature_settings,
            args.matcher_settings,
            args.max_reprojection_error,
            args.min_track_length,
        )
    )


if __name__ == "__main__":
    main()
