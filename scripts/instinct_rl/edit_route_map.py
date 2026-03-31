"""Export a deterministic top-down parkour map and interactively draw a route."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
sys.path.append(os.path.join(os.getcwd(), "source", "instinctlab"))
sys.path.append(os.path.join(os.getcwd(), "source", "instinctlab", "instinctlab", "tasks", "parkour", "scripts"))


from route_map_tool import (  # noqa: E402
    RouteMapEditorState,
    build_tile_wall_edges_grid,
    compute_centered_terrain_origins,
    compute_env_step_dt,
    draw_waypoint_path,
    load_route_payload,
    normalize_mouse_button,
    plot_schematic_terrain,
    plot_topdown_terrain,
    render_schematic_terrain_map,
    render_topdown_terrain_map,
    save_route_payload,
)
from play_runtime import select_center_terrain_origin, validate_isaacsim_python_environment  # noqa: E402
from config_loading import load_logged_config  # noqa: E402
from play_route import compute_tile_wall_edges_from_generator_cfg  # noqa: E402


_ISAAC_APP_LAUNCHER = None
_ISAAC_SIM_APP = None


def _ensure_isaac_app_launched(*, launcher_factory=None, validate_env_fn=None):
    global _ISAAC_APP_LAUNCHER, _ISAAC_SIM_APP

    if _ISAAC_SIM_APP is not None:
        return _ISAAC_SIM_APP

    accepted_eula = os.environ.get("OMNI_KIT_ACCEPT_EULA", "")
    if accepted_eula.strip().upper() not in {"YES", "Y", "TRUE", "1"}:
        raise RuntimeError(
            "Real mesh export requires Isaac Sim standalone bootstrap. Re-run with"
            " OMNI_KIT_ACCEPT_EULA=YES so the script can launch the headless Isaac app."
        )

    if validate_env_fn is None:
        validate_env_fn = validate_isaacsim_python_environment
    validate_env_fn()

    if launcher_factory is None:
        from isaaclab.app import AppLauncher

        launcher_factory = lambda: AppLauncher(headless=True)

    app_launcher = launcher_factory()
    _ISAAC_APP_LAUNCHER = app_launcher
    _ISAAC_SIM_APP = getattr(app_launcher, "app", app_launcher)
    return _ISAAC_SIM_APP


def _close_isaac_app_if_needed() -> None:
    global _ISAAC_APP_LAUNCHER, _ISAAC_SIM_APP

    if _ISAAC_SIM_APP is None:
        return

    close = getattr(_ISAAC_SIM_APP, "close", None)
    if callable(close):
        close()

    _ISAAC_SIM_APP = None
    _ISAAC_APP_LAUNCHER = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a seeded top-down parkour map and click waypoints into route.json.")
    parser.add_argument("--task", type=str, required=True, help="Task name, for example Instinct-Parkour-Target-Amp-G1-v0.")
    parser.add_argument("--seed", type=int, required=True, help="Terrain seed to export and encode into the route artifact.")
    parser.add_argument("--output_png", type=str, default=None, help="Output PNG path for the exported top-down map.")
    parser.add_argument("--output_route", type=str, default=None, help="Output route JSON path.")
    parser.add_argument("--load_route", type=str, default=None, help="Optional existing route JSON to edit.")
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Optional logged run directory name or absolute path used for the fallback schematic map source.",
    )
    parser.add_argument(
        "--waypoint_spacing_m",
        type=float,
        default=0.15,
        help="Metadata value stored into route.json for compatibility with recorded routes.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG export DPI.")
    parser.add_argument("--export_only", action="store_true", default=False, help="Export the PNG and exit without opening the editor.")
    parser.add_argument(
        "--real_mesh",
        action="store_true",
        default=False,
        help="Launch Isaac Sim and export the real mesh map instead of the default schematic route editor map.",
    )
    return parser


class RouteMapEditorSession:
    def __init__(
        self,
        *,
        figure,
        axis,
        terrain_mesh,
        terrain_origins,
        center_origin,
        editor_state: RouteMapEditorState,
        output_png: Path,
        output_route: Path,
        dpi: int,
        title: str,
        map_mode: str,
        terrain_generator_cfg: dict | None = None,
        tile_wall_edges=None,
    ):
        self.figure = figure
        self.axis = axis
        self.terrain_mesh = terrain_mesh
        self.terrain_origins = terrain_origins
        self.center_origin = center_origin
        self.editor_state = editor_state
        self.output_png = output_png
        self.output_route = output_route
        self.dpi = dpi
        self.title = title
        self.map_mode = map_mode
        self.terrain_generator_cfg = terrain_generator_cfg
        self.tile_wall_edges = tile_wall_edges
        self._status_artist = self.figure.text(0.02, 0.02, "", ha="left", va="bottom", fontsize=10)
        self._route_artist = None
        self._route_arrow_artists = []
        self._instructions_artist = self.figure.text(
            0.02,
            0.98,
            "Left click: add point | Right click / Backspace: undo | C: clear | Enter / S: save route | Esc: close",
            ha="left",
            va="top",
            fontsize=10,
        )
        self._refresh_overlay()
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_click(self, event) -> None:
        if event.inaxes != self.axis or event.xdata is None or event.ydata is None:
            return
        button = normalize_mouse_button(event.button)
        if button == "left":
            self.editor_state.add_waypoint(event.xdata, event.ydata)
        elif button == "right":
            self.editor_state.undo_last_waypoint()
        else:
            return
        self._refresh_overlay()

    def on_key_press(self, event) -> None:
        if event.key in {"backspace", "delete", "u"}:
            self.editor_state.undo_last_waypoint()
            self._refresh_overlay()
        elif event.key == "c":
            self.editor_state.clear_waypoints()
            self._refresh_overlay()
        elif event.key in {"enter", "s"}:
            self.save_and_close()
        elif event.key in {"escape", "q"}:
            plt.close(self.figure)

    def save_and_close(self) -> None:
        save_route_payload(self.output_route, self.editor_state.build_route_payload())
        if self.map_mode == "schematic":
            render_schematic_terrain_map(
                output_path=self.output_png,
                terrain_generator_cfg=self.terrain_generator_cfg,
                terrain_origins=self.terrain_origins,
                center_origin=self.center_origin,
                tile_wall_edges=self.tile_wall_edges,
                waypoints_xy=self.editor_state.waypoints_xy,
                title=self.title,
                dpi=self.dpi,
            )
        else:
            render_topdown_terrain_map(
                terrain_mesh=self.terrain_mesh,
                output_path=self.output_png,
                terrain_origins=self.terrain_origins,
                center_origin=self.center_origin,
                tile_wall_edges=self.tile_wall_edges,
                waypoints_xy=self.editor_state.waypoints_xy,
                title=self.title,
                dpi=self.dpi,
            )
        print(f"[INFO] Saved route JSON to {self.output_route}")
        print(f"[INFO] Saved annotated map PNG to {self.output_png}")
        plt.close(self.figure)

    def _refresh_overlay(self) -> None:
        if self._route_artist is not None:
            self._route_artist.remove()
            self._route_artist = None
        for artist in self._route_arrow_artists:
            artist.remove()
        self._route_arrow_artists = []
        if self.editor_state.waypoints_xy:
            self._route_artist, self._route_arrow_artists = draw_waypoint_path(
                self.axis,
                waypoints_xy=self.editor_state.waypoints_xy,
                anchor_xy=np.asarray(self.center_origin, dtype=np.float64).reshape(3)[:2],
                linewidth=2.0,
                markersize=4.5,
                arrow_mutation_scale=14.0,
            )
        self._status_artist.set_text(
            f"seed={self.editor_state.seed} | waypoints={len(self.editor_state.waypoints_xy)} | output={self.output_route.name}"
        )
        self.figure.canvas.draw_idle()


def attach_route_map_editor_session(**kwargs) -> RouteMapEditorSession:
    figure = kwargs["figure"]
    manager = getattr(figure.canvas, "manager", None)
    key_press_handler_id = getattr(manager, "key_press_handler_id", None)
    if key_press_handler_id is not None:
        figure.canvas.mpl_disconnect(key_press_handler_id)
    session = RouteMapEditorSession(**kwargs)
    setattr(figure, "_route_map_editor_session", session)
    return session


def _resolve_output_png(task: str, seed: int, output_png: str | None) -> Path:
    if output_png is not None:
        return Path(output_png)
    safe_task = task.replace("/", "_")
    return Path("outputs") / "parkour" / "maps" / f"{safe_task}-seed{seed}.png"


def _resolve_output_route(task: str, seed: int, output_route: str | None) -> Path:
    if output_route is not None:
        return Path(output_route)
    safe_task = task.replace("/", "_")
    return Path("outputs") / "parkour" / "routes" / f"{safe_task}-seed{seed}.json"


def _build_seeded_terrain(task: str, seed: int):
    _ensure_isaac_app_launched()
    import instinctlab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device="cpu", num_envs=1, use_fabric=False)
    env_cfg.seed = seed
    terrain_cfg = copy.deepcopy(env_cfg.scene.terrain.terrain_generator)
    if terrain_cfg is None:
        raise ValueError("This task does not expose a terrain generator.")
    terrain_cfg.seed = seed
    terrain_generator_cls = getattr(terrain_cfg, "class_type")
    terrain_generator = terrain_generator_cls(cfg=terrain_cfg, device="cpu")
    terrain_mesh = terrain_generator.terrain_mesh
    terrain_origins = terrain_generator.terrain_origins
    center_origin = select_center_terrain_origin(terrain_origins)
    step_dt = compute_env_step_dt(env_cfg)
    tile_wall_edges = build_tile_wall_edges_grid(
        subterrain_specific_cfgs=getattr(terrain_generator, "subterrain_specific_cfgs", None),
        num_rows=int(terrain_cfg.num_rows),
        num_cols=int(terrain_cfg.num_cols),
    )
    return terrain_mesh, terrain_origins, center_origin, step_dt, tile_wall_edges


def _resolve_logged_run_dir(load_run: str | None) -> Path:
    if load_run is not None:
        run_path = Path(load_run)
        if run_path.is_absolute():
            if not (run_path / "params" / "env.yaml").is_file():
                raise FileNotFoundError(f"Could not find params/env.yaml under {run_path}")
            return run_path
        candidates = sorted(Path("logs/instinct_rl").glob(f"*/{load_run}"))
        for candidate in candidates:
            if (candidate / "params" / "env.yaml").is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve logged run '{load_run}' under logs/instinct_rl.")

    env_yaml_paths = sorted(
        Path("logs/instinct_rl").glob("*/*/params/env.yaml"),
        key=lambda path: path.parent.parent.name,
    )
    if not env_yaml_paths:
        raise FileNotFoundError("Could not find any logged env.yaml under logs/instinct_rl.")
    return env_yaml_paths[-1].parent.parent


def _build_schematic_terrain_from_log(log_dir: Path, seed: int):
    env_cfg = load_logged_config(log_dir, "env")
    terrain_generator_cfg = copy.deepcopy(env_cfg["scene"]["terrain"]["terrain_generator"])
    terrain_generator_cfg["seed"] = seed
    terrain_origins = compute_centered_terrain_origins(
        num_rows=int(terrain_generator_cfg["num_rows"]),
        num_cols=int(terrain_generator_cfg["num_cols"]),
        tile_size=tuple(terrain_generator_cfg["size"]),
    )
    center_origin = select_center_terrain_origin(terrain_origins)
    step_dt = compute_env_step_dt(env_cfg)
    tile_wall_edges = compute_tile_wall_edges_from_generator_cfg(terrain_generator_cfg, seed)
    return terrain_generator_cfg, terrain_origins, center_origin, step_dt, tile_wall_edges


def _build_editor_state(
    *,
    task: str,
    seed: int,
    step_dt: float,
    waypoint_spacing_m: float,
    tile_wall_edges,
    load_route: str | None,
) -> RouteMapEditorState:
    if load_route is not None:
        editor_state = RouteMapEditorState.from_route_payload(load_route_payload(load_route))
        editor_state.seed = seed
        editor_state.task = task
        editor_state.step_dt = step_dt
        editor_state.waypoint_spacing_m = waypoint_spacing_m
        if tile_wall_edges is not None:
            editor_state.tile_wall_edges = tile_wall_edges
        return editor_state

    return RouteMapEditorState(
        task=task,
        seed=seed,
        step_dt=step_dt,
        waypoint_spacing_m=waypoint_spacing_m,
        tile_wall_edges=tile_wall_edges,
    )


def main() -> int:
    try:
        parser = build_arg_parser()
        args = parser.parse_args()

        terrain_generator_cfg = None
        tile_wall_edges = None
        if args.real_mesh:
            map_mode = "mesh"
            try:
                terrain_mesh, terrain_origins, center_origin, step_dt, tile_wall_edges = _build_seeded_terrain(
                    args.task, args.seed
                )
            except ModuleNotFoundError as exc:
                if exc.name not in {"pxr", "omni", "carb"}:
                    raise
                log_dir = _resolve_logged_run_dir(args.load_run)
                terrain_generator_cfg, terrain_origins, center_origin, step_dt, tile_wall_edges = (
                    _build_schematic_terrain_from_log(log_dir, args.seed)
                )
                terrain_mesh = None
                map_mode = "schematic"
                print(
                    f"[WARN] Falling back to schematic parkour map because dependency '{exc.name}' is unavailable."
                    f" Using logged env config from {log_dir}."
                )
        else:
            map_mode = "schematic"
            log_dir = _resolve_logged_run_dir(args.load_run)
            terrain_generator_cfg, terrain_origins, center_origin, step_dt, tile_wall_edges = _build_schematic_terrain_from_log(
                log_dir, args.seed
            )
            terrain_mesh = None
            print(
                f"[INFO] Using schematic parkour map from logged env config at {log_dir}."
                " Wall overlays are precomputed offline from the selected seed."
            )
        output_png = _resolve_output_png(args.task, args.seed, args.output_png)
        output_route = _resolve_output_route(args.task, args.seed, args.output_route)
        title = f"{args.task} | seed={args.seed}"

        if map_mode == "schematic":
            render_schematic_terrain_map(
                output_path=output_png,
                terrain_generator_cfg=terrain_generator_cfg,
                terrain_origins=terrain_origins,
                center_origin=center_origin,
                tile_wall_edges=tile_wall_edges,
                title=title,
                dpi=args.dpi,
            )
        else:
            render_topdown_terrain_map(
                terrain_mesh=terrain_mesh,
                output_path=output_png,
                terrain_origins=terrain_origins,
                center_origin=center_origin,
                tile_wall_edges=tile_wall_edges,
                title=title,
                dpi=args.dpi,
            )
        print(f"[INFO] Saved base map PNG to {output_png}")

        editor_state = _build_editor_state(
            task=args.task,
            seed=args.seed,
            step_dt=step_dt,
            waypoint_spacing_m=args.waypoint_spacing_m,
            tile_wall_edges=tile_wall_edges,
            load_route=args.load_route,
        )

        if args.export_only:
            save_route_payload(output_route, editor_state.build_route_payload())
            print(f"[INFO] Saved route JSON to {output_route}")
            return 0

        figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
        if map_mode == "schematic":
            plot_schematic_terrain(
                axis,
                terrain_generator_cfg=terrain_generator_cfg,
                terrain_origins=terrain_origins,
                center_origin=center_origin,
                tile_wall_edges=tile_wall_edges,
                waypoints_xy=editor_state.waypoints_xy,
                title=title,
            )
        else:
            plot_topdown_terrain(
                axis,
                terrain_mesh=terrain_mesh,
                terrain_origins=terrain_origins,
                center_origin=center_origin,
                tile_wall_edges=tile_wall_edges,
                waypoints_xy=editor_state.waypoints_xy,
                title=title,
            )
        session = attach_route_map_editor_session(
            figure=figure,
            axis=axis,
            terrain_mesh=terrain_mesh,
            terrain_origins=terrain_origins,
            center_origin=center_origin,
            editor_state=editor_state,
            output_png=output_png,
            output_route=output_route,
            dpi=args.dpi,
            title=title,
            map_mode=map_mode,
            terrain_generator_cfg=terrain_generator_cfg,
            tile_wall_edges=tile_wall_edges,
        )
        _ = session
        plt.show()
        return 0
    finally:
        _close_isaac_app_if_needed()


if __name__ == "__main__":
    raise SystemExit(main())
