from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import yaml
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


def _translate_wall_edges(
    wall_edges: list[dict] | None,
    *,
    delta_x: float,
    delta_y: float,
) -> list[dict]:
    if not wall_edges:
        return []
    return [
        {
            "side": wall["side"],
            "xy": (
                float(wall["xy"][0]) + float(delta_x),
                float(wall["xy"][1]) + float(delta_y),
            ),
            "width": float(wall["width"]),
            "height": float(wall["height"]),
        }
        for wall in wall_edges
    ]


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        super().__init__(cfg, device)

    def _resolve_precomputed_wall_edges_for_tile(self, *, row: int | None, col: int | None, size) -> list[dict] | None:
        tile_wall_edges = getattr(self.cfg, "precomputed_tile_wall_edges", None)
        if tile_wall_edges is None or row is None or col is None:
            return None
        if row >= len(tile_wall_edges) or col >= len(tile_wall_edges[row]):
            return None
        return _translate_wall_edges(
            tile_wall_edges[row][col],
            delta_x=float(size[0]) * 0.5,
            delta_y=float(size[1]) * 0.5,
        )

    def _get_terrain_mesh_for_tile(self, difficulty: float, cfg: SubTerrainBaseCfg, *, row: int | None, col: int | None):
        """Generate a sub-terrain mesh while preserving the realized per-tile wall metadata."""
        from isaaclab.utils.dict import dict_to_md5_hash
        from isaaclab.utils.io import dump_yaml

        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        precomputed_wall_edges = self._resolve_precomputed_wall_edges_for_tile(row=row, col=col, size=cfg.size)
        if precomputed_wall_edges is not None:
            setattr(cfg, "precomputed_wall_edges", precomputed_wall_edges)

        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            if os.path.exists(sub_terrain_meta_filename):
                with open(sub_terrain_meta_filename, encoding="utf-8") as file:
                    cache_metadata = yaml.full_load(file) or {}
                cached_wall_edges = cache_metadata.get("generated_wall_edges")
                if cached_wall_edges is not None:
                    setattr(cfg, "generated_wall_edges", cached_wall_edges)
            self._subterrain_specific_cfgs.append(cfg)
            return mesh, origin

        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)

        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]

        setattr(
            cfg,
            "generated_wall_edges",
            _translate_wall_edges(
                getattr(cfg, "generated_wall_edges", None),
                delta_x=float(transform[0, 3]),
                delta_y=float(transform[1, 3]),
            ),
        )

        if self.cfg.use_cache:
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)

        self._subterrain_specific_cfgs.append(cfg)
        return mesh, origin

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        return self._get_terrain_mesh_for_tile(difficulty, cfg, row=None, col=None)

    def _generate_random_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh_for_tile(
                difficulty,
                sub_terrains_cfgs[sub_index],
                row=int(sub_row),
                col=int(sub_col),
            )
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh_for_tile(
                    difficulty,
                    sub_terrains_cfgs[sub_indices[sub_col]],
                    row=int(sub_row),
                    col=int(sub_col),
                )
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
        return None
