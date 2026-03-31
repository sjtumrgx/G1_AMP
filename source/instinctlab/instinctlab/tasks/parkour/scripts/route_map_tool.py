from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.tri import Triangulation
import numpy as np


ROUTE_ARTIFACT_VERSION = 1


def compute_env_step_dt(env_cfg) -> float:
    if isinstance(env_cfg, dict):
        decimation = float(env_cfg["decimation"])
        sim_dt = float(env_cfg["sim"]["dt"])
    else:
        decimation = float(getattr(env_cfg, "decimation"))
        sim_dt = float(getattr(env_cfg.sim, "dt"))
    return decimation * sim_dt


class RouteMapEditorState:
    def __init__(
        self,
        *,
        task: str | None,
        seed: int | None,
        step_dt: float,
        waypoint_spacing_m: float = 0.15,
        waypoints_xy: list[list[float]] | None = None,
        tile_wall_edges=None,
    ):
        self.task = task
        self.seed = seed
        self.step_dt = float(step_dt)
        self.waypoint_spacing_m = float(waypoint_spacing_m)
        self.waypoints_xy: list[list[float]] = [list(point) for point in (waypoints_xy or [])]
        self.tile_wall_edges = clone_tile_wall_edges(tile_wall_edges)

    def add_waypoint(self, x: float, y: float) -> None:
        self.waypoints_xy.append([float(x), float(y)])

    def undo_last_waypoint(self) -> None:
        if self.waypoints_xy:
            self.waypoints_xy.pop()

    def clear_waypoints(self) -> None:
        self.waypoints_xy.clear()

    def build_route_payload(self) -> dict:
        return {
            "version": ROUTE_ARTIFACT_VERSION,
            "task": self.task,
            "seed": self.seed,
            "step_dt": self.step_dt,
            "waypoint_spacing_m": self.waypoint_spacing_m,
            "waypoints_xy": [list(point) for point in self.waypoints_xy],
            "tile_wall_edges": clone_tile_wall_edges(self.tile_wall_edges),
        }

    @classmethod
    def from_route_payload(cls, payload: dict) -> "RouteMapEditorState":
        return cls(
            task=payload.get("task"),
            seed=payload.get("seed"),
            step_dt=float(payload["step_dt"]),
            waypoint_spacing_m=float(payload.get("waypoint_spacing_m", 0.15)),
            waypoints_xy=[
                [float(point[0]), float(point[1])]
                for point in payload.get("waypoints_xy", [])
            ],
            tile_wall_edges=payload.get("tile_wall_edges"),
        )


def save_route_payload(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def load_route_payload(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    payload["step_dt"] = float(payload["step_dt"])
    payload["waypoint_spacing_m"] = float(payload.get("waypoint_spacing_m", 0.15))
    payload["waypoints_xy"] = [
        [float(point[0]), float(point[1])]
        for point in payload.get("waypoints_xy", [])
    ]
    if payload.get("seed") is not None:
        payload["seed"] = int(payload["seed"])
    payload["tile_wall_edges"] = clone_tile_wall_edges(payload.get("tile_wall_edges"))
    return payload


def normalize_mouse_button(button) -> str | None:
    if button is None:
        return None
    if isinstance(button, str):
        lowered = button.lower()
        if lowered in {"left", "right"}:
            return lowered
        return None
    button_name = getattr(button, "name", None)
    if isinstance(button_name, str):
        lowered = button_name.lower()
        if lowered in {"left", "right"}:
            return lowered
    if button == 1:
        return "left"
    if button == 3:
        return "right"
    return None


def draw_waypoint_path(
    axis,
    *,
    waypoints_xy: list[list[float]] | None,
    anchor_xy=None,
    color: str = "#0f172a",
    linewidth: float = 1.8,
    markersize: float = 4.0,
    arrow_mutation_scale: float = 12.0,
):
    if not waypoints_xy:
        return None, []
    waypoints = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    if anchor_xy is not None:
        anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
        waypoints = np.concatenate([anchor[None, :], waypoints], axis=0)
    (line_artist,) = axis.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "-o",
        color=color,
        linewidth=linewidth,
        markersize=markersize,
        zorder=6,
    )
    arrow_artists = []
    for index in range(len(waypoints) - 1):
        start = waypoints[index]
        end = waypoints[index + 1]
        arrow = FancyArrowPatch(
            posA=(float(start[0]), float(start[1])),
            posB=(float(end[0]), float(end[1])),
            arrowstyle="-|>",
            mutation_scale=arrow_mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=10.0,
            shrinkB=8.0,
            zorder=7,
        )
        axis.add_patch(arrow)
        arrow_artists.append(arrow)
    return line_artist, arrow_artists


def render_topdown_terrain_map(
    *,
    terrain_mesh,
    output_path: str | Path,
    terrain_origins=None,
    center_origin=None,
    tile_wall_edges=None,
    waypoints_xy: list[list[float]] | None = None,
    title: str | None = None,
    figure_size: tuple[float, float] = (9.0, 9.0),
    dpi: int = 200,
) -> Path:
    figure = Figure(figsize=figure_size, tight_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    plot_topdown_terrain(
        axis,
        terrain_mesh=terrain_mesh,
        terrain_origins=terrain_origins,
        center_origin=center_origin,
        tile_wall_edges=tile_wall_edges,
        waypoints_xy=waypoints_xy,
        title=title,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=dpi)
    return output_file


def compute_centered_terrain_origins(*, num_rows: int, num_cols: int, tile_size: tuple[float, float]) -> np.ndarray:
    size_x, size_y = float(tile_size[0]), float(tile_size[1])
    origins = np.zeros((int(num_rows), int(num_cols), 3), dtype=np.float32)
    x_offset = size_x * float(num_rows) * 0.5
    y_offset = size_y * float(num_cols) * 0.5
    for row in range(int(num_rows)):
        for col in range(int(num_cols)):
            origins[row, col, 0] = (row + 0.5) * size_x - x_offset
            origins[row, col, 1] = (col + 0.5) * size_y - y_offset
    return origins


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


def render_schematic_terrain_map(
    *,
    output_path: str | Path,
    terrain_generator_cfg: dict,
    terrain_origins,
    center_origin=None,
    tile_wall_edges=None,
    waypoints_xy: list[list[float]] | None = None,
    title: str | None = None,
    figure_size: tuple[float, float] = (9.0, 9.0),
    dpi: int = 200,
) -> Path:
    figure = Figure(figsize=figure_size, tight_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    plot_schematic_terrain(
        axis,
        terrain_generator_cfg=terrain_generator_cfg,
        terrain_origins=terrain_origins,
        center_origin=center_origin,
        tile_wall_edges=tile_wall_edges,
        waypoints_xy=waypoints_xy,
        title=title,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=dpi)
    return output_file


def compute_tile_wall_edges(
    *,
    tile_origin_xy: tuple[float, float] | list[float],
    tile_size: tuple[float, float],
    wall_prob,
    wall_thickness: float,
    seed: int | None,
    row: int,
    col: int,
    tile_name: str,
) -> list[dict]:
    if wall_prob is None:
        return []
    tile_origin_xy = np.asarray(tile_origin_xy, dtype=np.float64).reshape(2)
    size_x, size_y = float(tile_size[0]), float(tile_size[1])
    thickness = max(float(wall_thickness) * 3.0, 0.12)
    min_x = float(tile_origin_xy[0] - size_x * 0.5)
    max_x = float(tile_origin_xy[0] + size_x * 0.5)
    min_y = float(tile_origin_xy[1] - size_y * 0.5)
    max_y = float(tile_origin_xy[1] + size_y * 0.5)
    wall_prob = [float(value) for value in wall_prob]
    rng_seed = int.from_bytes(
        hashlib.md5(f"{seed}:{row}:{col}:{tile_name}".encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    rng = np.random.default_rng(rng_seed)
    wall_specs = [
        ("left", wall_prob[0], (min_x - thickness * 0.5, min_y), thickness, size_y),
        ("right", wall_prob[1], (max_x - thickness * 0.5, min_y), thickness, size_y),
        ("front", wall_prob[2], (min_x, min_y - thickness * 0.5), size_x, thickness),
        ("back", wall_prob[3], (min_x, max_y - thickness * 0.5), size_x, thickness),
    ]
    active_walls: list[dict] = []
    for side, probability, xy, width, height in wall_specs:
        if probability <= 0.0:
            continue
        if probability >= 1.0 or float(rng.uniform()) < probability:
            active_walls.append(
                {
                    "side": side,
                    "xy": (float(xy[0]), float(xy[1])),
                    "width": float(width),
                    "height": float(height),
                }
            )
    return active_walls


def draw_tile_walls(
    axis,
    *,
    wall_edges: list[dict],
    color: str = "#050505",
    linewidth: float = 3.4,
):
    wall_artists = []
    for wall in wall_edges:
        x0, y0, x1, y1 = wall_edge_to_segment(wall)
        (artist,) = axis.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=5,
        )
        wall_artists.append(artist)
    return wall_artists


def clone_wall_edges(wall_edges: list[dict] | None) -> list[dict]:
    if not wall_edges:
        return []
    return [
        {
            "side": wall["side"],
            "xy": (float(wall["xy"][0]), float(wall["xy"][1])),
            "width": float(wall["width"]),
            "height": float(wall["height"]),
        }
        for wall in wall_edges
    ]


def clone_tile_wall_edges(tile_wall_edges) -> list[list[list[dict]]] | None:
    if tile_wall_edges is None:
        return None
    return [
        [
            clone_wall_edges(cell)
            for cell in row
        ]
        for row in tile_wall_edges
    ]


def offset_wall_edges(wall_edges: list[dict] | None, offset_xy) -> list[dict]:
    if not wall_edges:
        return []
    offset = np.asarray(offset_xy, dtype=np.float64).reshape(2)
    translated = clone_wall_edges(wall_edges)
    for wall in translated:
        wall["xy"] = (
            float(wall["xy"][0]) + float(offset[0]),
            float(wall["xy"][1]) + float(offset[1]),
        )
    return translated


def build_tile_wall_edges_grid(
    *,
    subterrain_specific_cfgs,
    num_rows: int,
    num_cols: int,
) -> list[list[list[dict]]]:
    tile_wall_edges = [[[] for _ in range(int(num_cols))] for _ in range(int(num_rows))]
    if subterrain_specific_cfgs is None:
        return tile_wall_edges

    max_tiles = int(num_rows) * int(num_cols)
    for index, cfg in enumerate(list(subterrain_specific_cfgs)[:max_tiles]):
        row, col = np.unravel_index(index, (int(num_rows), int(num_cols)))
        tile_wall_edges[row][col] = clone_wall_edges(getattr(cfg, "generated_wall_edges", None))
    return tile_wall_edges


def wall_edge_to_segment(wall: dict) -> tuple[float, float, float, float]:
    x = float(wall["xy"][0])
    y = float(wall["xy"][1])
    width = float(wall["width"])
    height = float(wall["height"])
    side = wall["side"]
    if side == "left" or side == "right":
        wall_x = x + width * 0.5
        return wall_x, y, wall_x, y + height
    wall_y = y + height * 0.5
    return x, wall_y, x + width, wall_y


def plot_topdown_terrain(
    axis,
    *,
    terrain_mesh,
    terrain_origins=None,
    center_origin=None,
    tile_wall_edges=None,
    waypoints_xy: list[list[float]] | None = None,
    title: str | None = None,
) -> None:
    vertices = np.asarray(terrain_mesh.vertices, dtype=np.float64)
    faces = np.asarray(terrain_mesh.faces, dtype=np.int64)
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles=faces)
    face_heights = vertices[faces, 2].mean(axis=1)
    axis.tripcolor(triangulation, facecolors=face_heights, shading="flat", cmap="terrain")

    if terrain_origins is not None:
        origins = np.asarray(terrain_origins, dtype=np.float64).reshape(-1, 3)
        axis.scatter(origins[:, 0], origins[:, 1], s=8, c="white", alpha=0.35, linewidths=0.0)

    if center_origin is not None:
        center = np.asarray(center_origin, dtype=np.float64).reshape(3)
        axis.scatter([center[0]], [center[1]], s=90, marker="*", c="#e11d48", edgecolors="black", linewidths=0.6)

    if terrain_origins is not None and tile_wall_edges is not None:
        origins = np.asarray(terrain_origins, dtype=np.float64)
        for row in range(origins.shape[0]):
            for col in range(origins.shape[1]):
                translated_edges = offset_wall_edges(tile_wall_edges[row][col], origins[row, col, :2])
                if translated_edges:
                    draw_tile_walls(axis, wall_edges=translated_edges, linewidth=2.2)

    draw_waypoint_path(
        axis,
        waypoints_xy=waypoints_xy,
        anchor_xy=None if center_origin is None else np.asarray(center_origin, dtype=np.float64).reshape(3)[:2],
    )

    bounds = np.asarray(terrain_mesh.bounds, dtype=np.float64)
    padding = max(float(bounds[1, 0] - bounds[0, 0]), float(bounds[1, 1] - bounds[0, 1])) * 0.03
    padding = max(padding, 0.5)
    axis.set_xlim(bounds[0, 0] - padding, bounds[1, 0] + padding)
    axis.set_ylim(bounds[0, 1] - padding, bounds[1, 1] + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.grid(False)
    if title is not None:
        axis.set_title(title)


def plot_schematic_terrain(
    axis,
    *,
    terrain_generator_cfg: dict,
    terrain_origins,
    center_origin=None,
    tile_wall_edges=None,
    waypoints_xy: list[list[float]] | None = None,
    title: str | None = None,
) -> None:
    tile_size = terrain_generator_cfg["size"]
    size_x, size_y = float(tile_size[0]), float(tile_size[1])
    origins = np.asarray(terrain_origins, dtype=np.float64)
    column_names = compute_curriculum_column_subterrain_names(
        sub_terrains=terrain_generator_cfg["sub_terrains"],
        num_cols=int(terrain_generator_cfg["num_cols"]),
    )
    unique_names = list(dict.fromkeys(column_names))
    cmap = plt.get_cmap("tab20")
    color_denom = max(1, len(unique_names) - 1)
    color_lookup = {
        name: cmap(index / color_denom)
        for index, name in enumerate(unique_names)
    }

    for row in range(origins.shape[0]):
        for col in range(origins.shape[1]):
            origin = origins[row, col]
            tile_name = column_names[col]
            tile_cfg = terrain_generator_cfg["sub_terrains"][tile_name]
            difficulty_alpha = 0.35 + 0.45 * ((row + 0.5) / origins.shape[0])
            facecolor = color_lookup[tile_name]
            rect = plt.Rectangle(
                (origin[0] - size_x * 0.5, origin[1] - size_y * 0.5),
                size_x,
                size_y,
                facecolor=(facecolor[0], facecolor[1], facecolor[2], difficulty_alpha),
                edgecolor=(1.0, 1.0, 1.0, 0.18),
                linewidth=0.8,
            )
            axis.add_patch(rect)
            if tile_wall_edges is not None:
                translated_edges = offset_wall_edges(tile_wall_edges[row][col], origin[:2])
                if translated_edges:
                    draw_tile_walls(axis, wall_edges=translated_edges)
            axis.text(
                origin[0],
                origin[1],
                tile_name.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=6,
                color="#111827",
                alpha=0.8,
            )

    if center_origin is not None:
        center = np.asarray(center_origin, dtype=np.float64).reshape(3)
        axis.scatter([center[0]], [center[1]], s=90, marker="*", c="#e11d48", edgecolors="black", linewidths=0.6)

    draw_waypoint_path(
        axis,
        waypoints_xy=waypoints_xy,
        anchor_xy=None if center_origin is None else np.asarray(center_origin, dtype=np.float64).reshape(3)[:2],
    )

    bounds = np.asarray(
        [
            [origins[..., 0].min() - size_x * 0.5, origins[..., 1].min() - size_y * 0.5],
            [origins[..., 0].max() + size_x * 0.5, origins[..., 1].max() + size_y * 0.5],
        ],
        dtype=np.float64,
    )
    padding = max(float(bounds[1, 0] - bounds[0, 0]), float(bounds[1, 1] - bounds[0, 1])) * 0.03
    padding = max(padding, 0.5)
    axis.set_xlim(bounds[0, 0] - padding, bounds[1, 0] + padding)
    axis.set_ylim(bounds[0, 1] - padding, bounds[1, 1] + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.grid(False)
    if title is not None:
        axis.set_title(title)
