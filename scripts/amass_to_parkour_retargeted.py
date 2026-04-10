from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PARKOUR_MOTION_REFERENCE_MODEL_PATH = (
    REPO_ROOT
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "assets"
    / "resources"
    / "unitree_g1"
    / "urdf"
    / "g1_29dof_torsobase_popsicle.urdf"
)
PARKOUR_SCENE_ROBOT_MODEL_PATH = (
    REPO_ROOT
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "urdf"
    / "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)
DEFAULT_BASE_BODY_NAME = "torso_link"


def resolve_parkour_target_model_path() -> Path:
    """Return the URDF used by the parkour motion-reference kinematics."""
    return PARKOUR_MOTION_REFERENCE_MODEL_PATH


def resolve_parkour_scene_robot_model_path() -> Path:
    """Return the scene robot URDF used by Instinct-Parkour-Target-Amp-G1-v0."""
    return PARKOUR_SCENE_ROBOT_MODEL_PATH


def convert_motion_file(
    source_path: str | Path,
    *,
    base_body_name: str = DEFAULT_BASE_BODY_NAME,
) -> dict[str, np.ndarray]:
    """Convert a full-body stage2 motion file into the retargetted format used by parkour AMP."""
    source_path = Path(source_path)
    source = np.load(source_path, allow_pickle=True)

    required_keys = {
        "fps",
        "dof_names",
        "dof_positions",
        "body_names",
        "body_positions",
        "body_rotations",
    }
    missing_keys = sorted(required_keys.difference(source.files))
    if missing_keys:
        raise KeyError(f"Missing keys in {source_path}: {missing_keys}")

    body_names = source["body_names"].tolist()
    if base_body_name not in body_names:
        raise ValueError(f"Base body `{base_body_name}` not found in {source_path}")
    base_body_index = body_names.index(base_body_name)

    fps = float(np.asarray(source["fps"]).item())
    return {
        "framerate": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(source["dof_names"]),
        "joint_pos": np.asarray(source["dof_positions"], dtype=np.float32),
        "base_pos_w": np.asarray(source["body_positions"][:, base_body_index, :], dtype=np.float32),
        "base_quat_w": np.asarray(source["body_rotations"][:, base_body_index, :], dtype=np.float32),
    }


def write_motion_file(output_path: str | Path, motion_data: dict[str, np.ndarray]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **motion_data)


def iter_source_files(source_path: Path) -> Iterable[Path]:
    if source_path.is_file():
        yield source_path
        return
    for motion_path in sorted(source_path.rglob("*.npz")):
        yield motion_path


def build_output_path(source_path: Path, src_root: Path, dst_root: Path) -> Path:
    if src_root.is_file():
        return dst_root
    relative_path = source_path.relative_to(src_root)
    return dst_root / relative_path.parent / f"{source_path.stem}_retargetted.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AMASS_Retargeted_for_G1 full-body NPZ motions into Instinct parkour retargetted motions."
    )
    parser.add_argument("--src", type=Path, required=True, help="Input NPZ file or directory.")
    parser.add_argument("--dst", type=Path, required=True, help="Output NPZ file or directory.")
    parser.add_argument(
        "--base-body",
        type=str,
        default=DEFAULT_BASE_BODY_NAME,
        help="Body name to use as base_pos_w/base_quat_w. Defaults to torso_link for parkour AMP.",
    )
    parser.add_argument(
        "--print-target-model",
        action="store_true",
        help="Print the parkour motion-reference URDF path before converting.",
    )
    parser.add_argument(
        "--print-scene-model",
        action="store_true",
        help="Print the scene robot URDF path used by Instinct-Parkour-Target-Amp-G1-v0.",
    )
    args = parser.parse_args()

    if args.print_target_model:
        print(resolve_parkour_target_model_path())
    if args.print_scene_model:
        print(resolve_parkour_scene_robot_model_path())

    source_files = list(iter_source_files(args.src))
    if not source_files:
        raise FileNotFoundError(f"No .npz files found under {args.src}")

    for source_path in source_files:
        converted = convert_motion_file(source_path, base_body_name=args.base_body)
        output_path = build_output_path(source_path, args.src, args.dst)
        write_motion_file(output_path, converted)
        print(f"[OK] {source_path} -> {output_path}")


if __name__ == "__main__":
    main()
