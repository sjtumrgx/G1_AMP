from dataclasses import asdict
from dataclasses import dataclass
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import types

import numpy as np


ROUTE_ARTIFACT_VERSION = 1
_OFFLINE_TERRAIN_MODULES: dict[str, ModuleType] | None = None


@dataclass(frozen=True)
class RouteArtifact:
    version: int
    task: str | None
    seed: int | None
    step_dt: float
    waypoint_spacing_m: float
    waypoints_xy: list[list[float]]
    tile_wall_edges: list[list[list[dict]]] | None = None


def clone_wall_edge(wall: dict) -> dict:
    return {
        "side": str(wall["side"]),
        "xy": [float(wall["xy"][0]), float(wall["xy"][1])],
        "width": float(wall["width"]),
        "height": float(wall["height"]),
    }


def normalize_tile_wall_edges(tile_wall_edges) -> list[list[list[dict]]] | None:
    if tile_wall_edges is None:
        return None
    return [
        [
            [clone_wall_edge(wall) for wall in (cell or [])]
            for cell in row
        ]
        for row in tile_wall_edges
    ]


def build_tile_wall_edges_grid(*, subterrain_specific_cfgs, num_rows: int, num_cols: int) -> list[list[list[dict]]]:
    tile_wall_edges = [[[] for _ in range(int(num_cols))] for _ in range(int(num_rows))]
    if subterrain_specific_cfgs is None:
        return tile_wall_edges

    max_tiles = int(num_rows) * int(num_cols)
    for index, cfg in enumerate(list(subterrain_specific_cfgs)[:max_tiles]):
        row, col = np.unravel_index(index, (int(num_rows), int(num_cols)))
        tile_wall_edges[row][col] = normalize_tile_wall_edges([[[wall for wall in getattr(cfg, "generated_wall_edges", [])]]])[0][0]
    return tile_wall_edges


def compute_curriculum_column_subterrain_names(*, sub_terrains: dict[str, dict], num_cols: int) -> list[str]:
    if num_cols <= 0:
        return []
    names = list(sub_terrains.keys())
    proportions = np.array([float(sub_terrains[name]["proportion"]) for name in names], dtype=np.float64)
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)
    assigned: list[str] = []
    for index in range(int(num_cols)):
        sub_index = int(np.min(np.where(index / int(num_cols) + 0.001 < cumulative)[0]))
        assigned.append(names[sub_index])
    return assigned


def _deep_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _deep_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_deep_namespace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_namespace(item) for item in value)
    return value


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_offline_terrain_modules() -> dict[str, ModuleType]:
    global _OFFLINE_TERRAIN_MODULES
    if _OFFLINE_TERRAIN_MODULES is not None:
        return _OFFLINE_TERRAIN_MODULES

    repo_root = Path(__file__).resolve().parents[6]
    isaac_hf_utils_path = repo_root / ".." / "IsaacLab" / "source" / "isaaclab" / "isaaclab" / "terrains" / "height_field" / "utils.py"
    instinct_perlin_path = repo_root / "source" / "instinctlab" / "instinctlab" / "utils" / "perlin.py"
    hf_root = repo_root / "source" / "instinctlab" / "instinctlab" / "terrains" / "height_field"
    trimesh_root = repo_root / "source" / "instinctlab" / "instinctlab" / "terrains" / "trimesh"

    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab")).__dict__.setdefault("__path__", [])
    sys.modules.setdefault("isaaclab.terrains", types.ModuleType("isaaclab.terrains")).__dict__.setdefault("__path__", [])
    sys.modules.setdefault("isaaclab.terrains.height_field", types.ModuleType("isaaclab.terrains.height_field")).__dict__.setdefault("__path__", [])
    sys.modules.setdefault("instinctlab", types.ModuleType("instinctlab")).__dict__.setdefault("__path__", [])
    sys.modules.setdefault("instinctlab.utils", types.ModuleType("instinctlab.utils")).__dict__.setdefault("__path__", [])

    modules: dict[str, ModuleType] = {}
    modules["isaac_hf_utils"] = _load_module_from_path(
        "isaaclab.terrains.height_field.utils",
        isaac_hf_utils_path.resolve(),
    )
    modules["instinct_perlin"] = _load_module_from_path(
        "instinctlab.utils.perlin",
        instinct_perlin_path.resolve(),
    )

    parent_pkg = types.ModuleType("parkour_offline_terrains")
    parent_pkg.__path__ = [str((repo_root / "source" / "instinctlab" / "instinctlab" / "terrains").resolve())]
    sys.modules["parkour_offline_terrains"] = parent_pkg

    hf_pkg = types.ModuleType("parkour_offline_terrains.height_field")
    hf_pkg.__path__ = [str(hf_root.resolve())]
    sys.modules["parkour_offline_terrains.height_field"] = hf_pkg
    sys.modules["parkour_offline_terrains.height_field.hf_terrains_cfg"] = ModuleType(
        "parkour_offline_terrains.height_field.hf_terrains_cfg"
    )
    modules["hf_utils"] = _load_module_from_path(
        "parkour_offline_terrains.height_field.utils",
        (hf_root / "utils.py").resolve(),
    )
    modules["hf_terrains"] = _load_module_from_path(
        "parkour_offline_terrains.height_field.hf_terrains",
        (hf_root / "hf_terrains.py").resolve(),
    )

    tm_pkg = types.ModuleType("parkour_offline_terrains.trimesh")
    tm_pkg.__path__ = [str(trimesh_root.resolve())]
    sys.modules["parkour_offline_terrains.trimesh"] = tm_pkg
    sys.modules["parkour_offline_terrains.trimesh.mesh_terrains_cfg"] = ModuleType(
        "parkour_offline_terrains.trimesh.mesh_terrains_cfg"
    )
    modules["tm_utils"] = _load_module_from_path(
        "parkour_offline_terrains.trimesh.utils",
        (trimesh_root / "utils.py").resolve(),
    )
    modules["tm_terrains"] = _load_module_from_path(
        "parkour_offline_terrains.trimesh.mesh_terrains",
        (trimesh_root / "mesh_terrains.py").resolve(),
    )

    _OFFLINE_TERRAIN_MODULES = modules
    return modules


def _resolve_offline_terrain_function(function_ref: str):
    modules = _ensure_offline_terrain_modules()
    module_name, function_name = function_ref.split(":")
    if module_name == "instinctlab.terrains.height_field.hf_terrains":
        return getattr(modules["hf_terrains"], function_name)
    if module_name == "instinctlab.terrains.trimesh.mesh_terrains":
        return getattr(modules["tm_terrains"], function_name)
    raise ValueError(f"Unsupported offline terrain function: {function_ref}")


def _translate_wall_edges(tile_wall_edges: list[dict], *, dx: float, dy: float) -> list[dict]:
    return [
        {
            "side": wall["side"],
            "xy": [float(wall["xy"][0]) + float(dx), float(wall["xy"][1]) + float(dy)],
            "width": float(wall["width"]),
            "height": float(wall["height"]),
        }
        for wall in tile_wall_edges
    ]


def _compute_single_tile_wall_edges(subterrain_cfg: dict, difficulty: float, seed: int) -> list[dict]:
    terrain_cfg = copy.deepcopy(subterrain_cfg)
    terrain_fn = _resolve_offline_terrain_function(terrain_cfg["function"])
    terrain_cfg["seed"] = seed
    cfg = _deep_namespace(terrain_cfg)
    terrain_fn(float(difficulty), cfg)
    generated_wall_edges = getattr(cfg, "generated_wall_edges", [])
    return _translate_wall_edges(
        [clone_wall_edge(wall) for wall in generated_wall_edges],
        dx=-float(cfg.size[0]) * 0.5,
        dy=-float(cfg.size[1]) * 0.5,
    )


def compute_tile_wall_edges_from_generator_cfg(terrain_generator_cfg: dict, seed: int) -> list[list[list[dict]]]:
    generator_cfg = copy.deepcopy(terrain_generator_cfg)
    generator_cfg["seed"] = int(seed)
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    num_rows = int(generator_cfg["num_rows"])
    num_cols = int(generator_cfg["num_cols"])
    lower, upper = generator_cfg.get("difficulty_range", (0.0, 1.0))
    sub_terrains = generator_cfg["sub_terrains"]
    tile_wall_edges = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    if bool(generator_cfg.get("curriculum", False)):
        column_names = compute_curriculum_column_subterrain_names(sub_terrains=sub_terrains, num_cols=num_cols)
        for col in range(num_cols):
            tile_name = column_names[col]
            tile_cfg = sub_terrains[tile_name]
            for row in range(num_rows):
                difficulty = lower + (upper - lower) * ((row + float(rng.uniform())) / num_rows)
                tile_wall_edges[row][col] = _compute_single_tile_wall_edges(tile_cfg, difficulty, int(seed))
        return tile_wall_edges

    names = list(sub_terrains.keys())
    proportions = np.array([float(sub_terrains[name]["proportion"]) for name in names], dtype=np.float64)
    proportions /= proportions.sum()
    for index in range(num_rows * num_cols):
        row, col = np.unravel_index(index, (num_rows, num_cols))
        tile_name = names[int(rng.choice(len(names), p=proportions))]
        difficulty = float(rng.uniform(lower, upper))
        tile_wall_edges[row][col] = _compute_single_tile_wall_edges(sub_terrains[tile_name], difficulty, int(seed))
    return tile_wall_edges


class RouteWaypointRecorder:
    def __init__(
        self,
        *,
        task: str | None,
        seed: int | None,
        step_dt: float,
        waypoint_spacing_m: float,
        tile_wall_edges: list[list[list[dict]]] | None = None,
    ):
        self.task = task
        self.seed = seed
        self.step_dt = float(step_dt)
        self.waypoint_spacing_m = float(waypoint_spacing_m)
        self.tile_wall_edges = normalize_tile_wall_edges(tile_wall_edges)
        self._waypoints_xy: list[list[float]] = []

    def record_position(self, position_w) -> None:
        xy = np.asarray(position_w, dtype=np.float64).reshape(-1)[:2]
        if xy.shape[0] != 2:
            raise ValueError(f"Expected at least 2 position values, got shape {xy.shape}.")
        waypoint = [float(xy[0]), float(xy[1])]
        if not self._waypoints_xy:
            self._waypoints_xy.append(waypoint)
            return
        previous = np.asarray(self._waypoints_xy[-1], dtype=np.float64)
        if np.linalg.norm(xy - previous) >= self.waypoint_spacing_m:
            self._waypoints_xy.append(waypoint)

    def build_artifact(self) -> RouteArtifact:
        return RouteArtifact(
            version=ROUTE_ARTIFACT_VERSION,
            task=self.task,
            seed=self.seed,
            step_dt=self.step_dt,
            waypoint_spacing_m=self.waypoint_spacing_m,
            waypoints_xy=[list(point) for point in self._waypoints_xy],
            tile_wall_edges=normalize_tile_wall_edges(self.tile_wall_edges),
        )


class RouteWaypointFollower:
    def __init__(
        self,
        *,
        waypoints_xy: list[list[float]] | list[tuple[float, float]],
        limits,
        lookahead_distance_m: float,
        goal_tolerance_m: float,
        cruise_speed: float | None = None,
        heading_gain: float = 1.5,
    ):
        if len(waypoints_xy) == 0:
            raise ValueError("Route replay requires at least one waypoint.")
        self.waypoints_xy = [np.asarray(point, dtype=np.float64).reshape(2) for point in waypoints_xy]
        self.limits = limits
        self.lookahead_distance_m = float(lookahead_distance_m)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.cruise_speed = (
            float(cruise_speed) if cruise_speed is not None else float(getattr(limits, "lin_vel_x_max"))
        )
        self.heading_gain = float(heading_gain)
        self._segment_vectors = [
            self.waypoints_xy[index + 1] - self.waypoints_xy[index]
            for index in range(len(self.waypoints_xy) - 1)
        ]
        self._segment_lengths = [float(np.linalg.norm(vector)) for vector in self._segment_vectors]
        self._cumulative_lengths = [0.0]
        for segment_length in self._segment_lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + segment_length)
        self._path_progress = 0.0
        self._path_length = self._cumulative_lengths[-1]
        self._is_complete = len(self.waypoints_xy) == 1 or self._path_length <= 1e-6

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    def compute_command(self, *, position_xy, yaw: float) -> np.ndarray:
        if self._is_complete:
            return np.zeros(3, dtype=np.float32)

        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        if np.linalg.norm(self.waypoints_xy[-1] - position) <= self.goal_tolerance_m:
            self._is_complete = True
            return np.zeros(3, dtype=np.float32)

        target = self._resolve_lookahead_target(position)
        delta = target - position
        desired_heading = math.atan2(float(delta[1]), float(delta[0]))
        heading_error = _wrap_to_pi(desired_heading - float(yaw))
        yaw_rate = float(
            np.clip(
                self.heading_gain * heading_error,
                float(getattr(self.limits, "ang_vel_z_min")),
                float(getattr(self.limits, "ang_vel_z_max")),
            )
        )
        forward_scale = max(0.0, math.cos(heading_error))
        forward_velocity = min(self.cruise_speed, float(getattr(self.limits, "lin_vel_x_max"))) * forward_scale
        forward_velocity = float(
            np.clip(
                forward_velocity,
                float(getattr(self.limits, "lin_vel_x_min")),
                float(getattr(self.limits, "lin_vel_x_max")),
            )
        )
        if abs(heading_error) > math.pi / 2.0:
            forward_velocity = 0.0
        return np.asarray([forward_velocity, 0.0, yaw_rate], dtype=np.float32)

    def _resolve_lookahead_target(self, position_xy: np.ndarray) -> np.ndarray:
        closest_progress = self._compute_closest_progress(position_xy)
        self._path_progress = max(self._path_progress, closest_progress)
        lookahead_progress = min(self._path_progress + self.lookahead_distance_m, self._path_length)
        return self._interpolate_progress(lookahead_progress).astype(np.float32)

    def _compute_closest_progress(self, position_xy: np.ndarray) -> float:
        if len(self.waypoints_xy) == 1:
            return 0.0
        best_distance_sq = float("inf")
        best_progress = self._path_progress
        for index, (start, vector, segment_length) in enumerate(
            zip(self.waypoints_xy[:-1], self._segment_vectors, self._segment_lengths, strict=True)
        ):
            if segment_length <= 1e-8:
                continue
            segment_length_sq = segment_length * segment_length
            local_t = float(np.clip(np.dot(position_xy - start, vector) / segment_length_sq, 0.0, 1.0))
            projection = start + local_t * vector
            distance_sq = float(np.sum((position_xy - projection) ** 2))
            progress = self._cumulative_lengths[index] + local_t * segment_length
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_progress = progress
        return best_progress

    def _interpolate_progress(self, progress: float) -> np.ndarray:
        if progress <= 0.0 or len(self.waypoints_xy) == 1:
            return self.waypoints_xy[0]
        if progress >= self._path_length:
            return self.waypoints_xy[-1]
        for index, segment_length in enumerate(self._segment_lengths):
            next_progress = self._cumulative_lengths[index + 1]
            if progress <= next_progress:
                if segment_length <= 1e-8:
                    return self.waypoints_xy[index + 1]
                local_distance = progress - self._cumulative_lengths[index]
                local_t = local_distance / segment_length
                return self.waypoints_xy[index] + local_t * self._segment_vectors[index]
        return self.waypoints_xy[-1]


def save_route_artifact(path: str | Path, artifact: RouteArtifact) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(artifact), file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def load_route_artifact(path: str | Path) -> RouteArtifact:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    waypoints_xy = [
        [float(point[0]), float(point[1])]
        for point in payload["waypoints_xy"]
    ]
    return RouteArtifact(
        version=int(payload.get("version", ROUTE_ARTIFACT_VERSION)),
        task=payload.get("task"),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        step_dt=float(payload["step_dt"]),
        waypoint_spacing_m=float(payload["waypoint_spacing_m"]),
        waypoints_xy=waypoints_xy,
        tile_wall_edges=normalize_tile_wall_edges(payload.get("tile_wall_edges")),
    )


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
