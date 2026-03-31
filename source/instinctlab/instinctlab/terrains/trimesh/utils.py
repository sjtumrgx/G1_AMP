from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg


def _record_generated_wall_edge(cfg, *, side: str, xy: tuple[float, float], width: float, height: float) -> None:
    wall_edges = getattr(cfg, "generated_wall_edges", None)
    if wall_edges is None:
        wall_edges = []
        setattr(cfg, "generated_wall_edges", wall_edges)
    wall_edges.append(
        {
            "side": side,
            "xy": (float(xy[0]), float(xy[1])),
            "width": float(width),
            "height": float(height),
        }
    )


def _build_wall_mesh_from_edge(*, wall: dict, wall_height: float) -> trimesh.Trimesh:
    width = float(wall["width"])
    height = float(wall["height"])
    x = float(wall["xy"][0])
    y = float(wall["xy"][1])
    return trimesh.creation.box(
        extents=[width, height, float(wall_height)],
        transform=trimesh.transformations.translation_matrix(
            [x + width * 0.5, y + height * 0.5, float(wall_height) * 0.5]
        ),
    )


def crop_terrain_mesh_aabb(
    terrain_mesh: trimesh.Trimesh,
    x_max: float | None = None,
    x_min: float | None = None,
    y_max: float | None = None,
    y_min: float | None = None,
    z_max: float | None = None,
    z_min: float | None = None,
) -> trimesh.Trimesh:
    """Crop the terrain mesh to given the bounding box coordinates.
    Args:
        terrain_mesh (trimesh.Trimesh): The terrain mesh to be cropped.
        size (tuple[float, float]): The size of the bounding box (width, depth).
    Returns:
        trimesh.Trimesh: The cropped terrain mesh.
    """

    if x_max is not None:
        slice_plane_normal = [-1, 0, 0]
        slice_plane_origin = [x_max, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if x_min is not None:
        slice_plane_normal = [1, 0, 0]
        slice_plane_origin = [x_min, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_max is not None:
        slice_plane_normal = [0, -1, 0]
        slice_plane_origin = [0, y_max, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_min is not None:
        slice_plane_normal = [0, 1, 0]
        slice_plane_origin = [0, y_min, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_max is not None:
        slice_plane_normal = [0, 0, -1]
        slice_plane_origin = [0, 0, z_max]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_min is not None:
        slice_plane_normal = [0, 0, 1]
        slice_plane_origin = [0, 0, z_min]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    return terrain_mesh


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        setattr(cfg, "generated_wall_edges", [])
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = meshes

        precomputed_wall_edges = getattr(cfg, "precomputed_wall_edges", None)
        if precomputed_wall_edges is not None:
            for wall in precomputed_wall_edges:
                result_meshes.append(_build_wall_mesh_from_edge(wall=wall, wall_height=wall_height))
                _record_generated_wall_edge(
                    cfg,
                    side=wall["side"],
                    xy=(float(wall["xy"][0]), float(wall["xy"][1])),
                    width=float(wall["width"]),
                    height=float(wall["height"]),
                )
            return result_meshes, origin

        # Get mesh bounds
        mesh = max(meshes, key=lambda m: np.prod(m.bounds[1, :2] - m.bounds[0, :2]))
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # Left wall
        if np.random.uniform() < cfg.wall_prob[0]:
            left_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(left_wall)
            _record_generated_wall_edge(
                cfg,
                side="left",
                xy=(min_bound[0] - wall_thickness, min_bound[1]),
                width=wall_thickness,
                height=max_bound[1] - min_bound[1],
            )

        # Right wall
        if np.random.uniform() < cfg.wall_prob[1]:
            right_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(right_wall)
            _record_generated_wall_edge(
                cfg,
                side="right",
                xy=(max_bound[0], min_bound[1]),
                width=wall_thickness,
                height=max_bound[1] - min_bound[1],
            )

        # Front wall
        if np.random.uniform() < cfg.wall_prob[2]:
            front_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(front_wall)
            _record_generated_wall_edge(
                cfg,
                side="front",
                xy=(min_bound[0], min_bound[1] - wall_thickness),
                width=max_bound[0] - min_bound[0],
                height=wall_thickness,
            )

        # Back wall
        if np.random.uniform() < cfg.wall_prob[3]:
            back_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(back_wall)
            _record_generated_wall_edge(
                cfg,
                side="back",
                xy=(min_bound[0], max_bound[1]),
                width=max_bound[0] - min_bound[0],
                height=wall_thickness,
            )

        return result_meshes, origin

    return wrapper
