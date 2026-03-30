# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import numpy as np

from parkour_mujoco_runtime import (
    DEFAULT_DEPTH_FRAME_SHAPE,
    KeyboardCommandState,
    MuJoCoDepthAdapter,
    ParkourProprioObservationBuilder,
    ZeroDepthAdapter,
    build_cli_parser,
    compute_projected_gravity,
    extract_actuated_joint_order_from_urdf,
    extract_model_joint_indices,
    handle_keyboard_command,
    load_logged_yaml,
    load_parkour_onnx_sessions,
    prepare_mujoco_model_xml,
    resolve_default_model_path,
    resolve_joint_init_targets,
    resolve_joint_action_scale,
    resolve_joint_armature,
    resolve_joint_delay_steps,
    resolve_joint_drive_gains,
    resolve_joint_effort_limits,
    resolve_logged_run_paths,
    resolve_nominal_velocity_command,
    resolve_policy_observation_scales,
    settle_root_to_floor,
    run_parkour_policy,
    validate_run_artifacts,
    DelayedPDController,
)


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    import mujoco

    run_paths = resolve_logged_run_paths(args.load_run)
    validate_run_artifacts(run_paths)
    env_cfg = load_logged_yaml(run_paths.env_cfg)

    default_model_path = resolve_default_model_path(run_paths)
    model_source_path = Path(args.mjcf).resolve() if args.mjcf is not None else default_model_path

    with tempfile.TemporaryDirectory(prefix="parkour_mujoco_") as temp_dir:
        model_xml_path = prepare_mujoco_model_xml(model_source_path, temp_dir)
        model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)

        policy_joint_order = extract_actuated_joint_order_from_urdf(default_model_path)
        qpos_indices, qvel_indices = extract_model_joint_indices(model, policy_joint_order)
        _initialize_sim_state(data, qpos_indices, policy_joint_order, env_cfg)
        default_joint_targets = resolve_joint_init_targets(policy_joint_order, env_cfg)
        action_scale = resolve_joint_action_scale(policy_joint_order, env_cfg)
        armature = resolve_joint_armature(policy_joint_order, env_cfg)
        stiffness, damping = resolve_joint_drive_gains(policy_joint_order, env_cfg)
        effort_limits = resolve_joint_effort_limits(policy_joint_order, env_cfg)
        observation_scales = resolve_policy_observation_scales(env_cfg)
        model.dof_armature[qvel_indices] = armature
        model.dof_damping[qvel_indices] = damping
        settle_root_to_floor(model, data)
        controller = DelayedPDController(
            stiffness=stiffness,
            damping=damping,
            delay_steps=resolve_joint_delay_steps(env_cfg),
            default_targets=default_joint_targets,
        )
        encoder_session, actor_session = load_parkour_onnx_sessions(run_paths.export_dir)

        command_state = KeyboardCommandState(command=resolve_nominal_velocity_command(env_cfg))
        observation_builder = ParkourProprioObservationBuilder(
            joint_count=len(policy_joint_order),
            history_length=8,
            observation_scales=observation_scales,
        )
        if args.depth_mode == "zeros":
            depth_adapter = ZeroDepthAdapter(history_length=8, frame_shape=DEFAULT_DEPTH_FRAME_SHAPE)
        else:
            depth_adapter = MuJoCoDepthAdapter(model=model, history_length=8, frame_shape=DEFAULT_DEPTH_FRAME_SHAPE)

        previous_action = np.zeros(len(policy_joint_order), dtype=np.float32)
        target_joint_qpos = default_joint_targets.copy()
        root_target_z = float(data.qpos[2])

        initial_components = _read_policy_inputs(
            data=data,
            qpos_indices=qpos_indices,
            qvel_indices=qvel_indices,
            default_joint_targets=default_joint_targets,
            velocity_command=command_state.command,
            previous_action=previous_action,
        )
        for _ in range(8):
            observation_builder.push(**initial_components)

        if isinstance(depth_adapter, MuJoCoDepthAdapter):
            for _ in range(8):
                depth_adapter.update(data)

        recorded_commands: list[np.ndarray] = []
        recorded_actions: list[np.ndarray] = []
        recorded_joint_pos: list[np.ndarray] = []
        recorded_joint_vel: list[np.ndarray] = []

        sim_steps = 0
        control_steps = 0
        max_sim_steps = _resolve_max_sim_steps(args.sim_duration, model.opt.timestep, args.headless)

        if args.headless:
            try:
                while sim_steps < (max_sim_steps or 0):
                    sim_steps, control_steps, previous_action, target_joint_qpos = _simulate_step(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        qpos_indices=qpos_indices,
                        qvel_indices=qvel_indices,
                        default_joint_targets=default_joint_targets,
                        action_scale=action_scale,
                        effort_limits=effort_limits,
                        controller=controller,
                        encoder_session=encoder_session,
                        actor_session=actor_session,
                        observation_builder=observation_builder,
                        depth_adapter=depth_adapter,
                        command_state=command_state,
                        previous_action=previous_action,
                        target_joint_qpos=target_joint_qpos,
                        sim_steps=sim_steps,
                        control_steps=control_steps,
                        zero_act_until=args.zero_act_until,
                        root_target_z=root_target_z,
                        recorded_commands=recorded_commands,
                        recorded_actions=recorded_actions,
                        recorded_joint_pos=recorded_joint_pos,
                        recorded_joint_vel=recorded_joint_vel,
                    )
            finally:
                if isinstance(depth_adapter, MuJoCoDepthAdapter):
                    depth_adapter.close()
        else:
            import mujoco.viewer

            def on_key(keycode: int) -> None:
                if not args.keyboard_control:
                    return
                try:
                    key = chr(keycode).lower()
                except ValueError:
                    return
                handle_keyboard_command(
                    key,
                    command_state,
                    linvel_step=args.keyboard_linvel_step,
                    angvel=args.keyboard_angvel,
                )

            with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer_handle:
                _configure_viewer_camera(viewer_handle)
                try:
                    while _should_continue(viewer_handle, sim_steps, max_sim_steps):
                        step_start = time.perf_counter()
                        sim_steps, control_steps, previous_action, target_joint_qpos = _simulate_step(
                            mujoco=mujoco,
                            model=model,
                            data=data,
                            qpos_indices=qpos_indices,
                            qvel_indices=qvel_indices,
                            default_joint_targets=default_joint_targets,
                            action_scale=action_scale,
                            effort_limits=effort_limits,
                            controller=controller,
                            encoder_session=encoder_session,
                            actor_session=actor_session,
                            observation_builder=observation_builder,
                            depth_adapter=depth_adapter,
                            command_state=command_state,
                            previous_action=previous_action,
                            target_joint_qpos=target_joint_qpos,
                            sim_steps=sim_steps,
                            control_steps=control_steps,
                            zero_act_until=args.zero_act_until,
                            root_target_z=root_target_z,
                            recorded_commands=recorded_commands,
                            recorded_actions=recorded_actions,
                            recorded_joint_pos=recorded_joint_pos,
                            recorded_joint_vel=recorded_joint_vel,
                        )
                        with viewer_handle.lock():
                            viewer_handle.sync()
                        elapsed = time.perf_counter() - step_start
                        sleep_s = model.opt.timestep - elapsed
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                finally:
                    if isinstance(depth_adapter, MuJoCoDepthAdapter):
                        depth_adapter.close()

        if args.record_npz is not None:
            np.savez(
                args.record_npz,
                commands=np.asarray(recorded_commands, dtype=np.float32),
                actions=np.asarray(recorded_actions, dtype=np.float32),
                joint_pos=np.asarray(recorded_joint_pos, dtype=np.float32),
                joint_vel=np.asarray(recorded_joint_vel, dtype=np.float32),
            )

    print(
        "Completed MuJoCo sim2sim smoke run"
        f" | sim_steps={sim_steps}"
        f" | control_steps={control_steps}"
        f" | model={model_source_path}"
        f" | depth_mode={args.depth_mode}"
    )
    return 0


def _read_policy_inputs(
    *,
    data,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    default_joint_targets: np.ndarray,
    velocity_command: np.ndarray,
    previous_action: np.ndarray,
) -> dict[str, np.ndarray]:
    root_quat = data.qpos[3:7] if data.qpos.shape[0] >= 7 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = data.qvel[3:6] if data.qvel.shape[0] >= 6 else np.zeros(3, dtype=np.float32)
    return {
        "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32),
        "projected_gravity": compute_projected_gravity(root_quat),
        "velocity_command": np.asarray(velocity_command, dtype=np.float32),
        "joint_pos_rel": np.asarray(data.qpos[qpos_indices] - default_joint_targets, dtype=np.float32),
        "joint_vel_rel": np.asarray(data.qvel[qvel_indices], dtype=np.float32),
        "action": np.asarray(previous_action, dtype=np.float32),
    }


def _resolve_max_sim_steps(sim_duration: float | None, sim_dt: float, headless: bool) -> int | None:
    if sim_duration is not None:
        return max(1, int(round(sim_duration / sim_dt)))
    if headless:
        return int(round(5.0 / sim_dt))
    return None


def _should_continue(viewer_handle, sim_steps: int, max_sim_steps: int | None) -> bool:
    if max_sim_steps is not None and sim_steps >= max_sim_steps:
        return False
    if viewer_handle is None:
        return True
    return viewer_handle.is_running()


def _initialize_sim_state(data, qpos_indices: np.ndarray, joint_order: list[str], env_cfg: dict) -> None:
    import re

    init_state_cfg = env_cfg["scene"]["robot"]["init_state"]
    data.qpos[:3] = np.asarray(init_state_cfg["pos"], dtype=np.float64)
    data.qpos[3:7] = np.asarray(init_state_cfg["rot"], dtype=np.float64)
    data.qvel[:] = 0.0
    for index, joint_name in enumerate(joint_order):
        joint_pos = 0.0
        for pattern, value in init_state_cfg["joint_pos"].items():
            if re.fullmatch(pattern, joint_name):
                joint_pos = float(value)
                break
        data.qpos[qpos_indices[index]] = joint_pos


def _simulate_step(
    *,
    mujoco,
    model,
    data,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    default_joint_targets: np.ndarray,
    action_scale: np.ndarray,
    effort_limits: np.ndarray,
    controller: DelayedPDController,
    encoder_session,
    actor_session,
    observation_builder: ParkourProprioObservationBuilder,
    depth_adapter,
    command_state: KeyboardCommandState,
    previous_action: np.ndarray,
    target_joint_qpos: np.ndarray,
    sim_steps: int,
    control_steps: int,
    zero_act_until: int,
    root_target_z: float,
    recorded_commands: list[np.ndarray],
    recorded_actions: list[np.ndarray],
    recorded_joint_pos: list[np.ndarray],
    recorded_joint_vel: list[np.ndarray],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    if sim_steps % 4 == 0:
        if isinstance(depth_adapter, MuJoCoDepthAdapter):
            depth_adapter.update(data)

        policy_inputs = _read_policy_inputs(
            data=data,
            qpos_indices=qpos_indices,
            qvel_indices=qvel_indices,
            default_joint_targets=default_joint_targets,
            velocity_command=command_state.command,
            previous_action=previous_action,
        )
        observation_builder.push(**policy_inputs)
        proprio_observation = observation_builder.build()
        depth_observation = depth_adapter.build()

        action = run_parkour_policy(
            encoder_session=encoder_session,
            actor_session=actor_session,
            proprio_observation=proprio_observation,
            depth_observation=depth_observation,
        )
        action = np.clip(action, -10.0, 10.0)
        if control_steps < zero_act_until:
            action[:] = 0.0

        target_joint_qpos = default_joint_targets + action_scale * action
        previous_action = action
        control_steps += 1

        recorded_commands.append(command_state.command.copy())
        recorded_actions.append(action.copy())
        recorded_joint_pos.append(data.qpos[qpos_indices].astype(np.float32, copy=True))
        recorded_joint_vel.append(data.qvel[qvel_indices].astype(np.float32, copy=True))

    torque = controller.compute_torque(
        target_joint_qpos,
        data.qpos[qpos_indices],
        data.qvel[qvel_indices],
    )
    torque = np.clip(torque, -effort_limits, effort_limits)
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[qvel_indices] = torque
    if command_state.elastic_band_enabled:
        z_error = root_target_z - float(data.qpos[2])
        data.qfrc_applied[2] += 800.0 * z_error - 120.0 * float(data.qvel[2])
        data.qfrc_applied[3] += -250.0 * float(data.qpos[4]) - 30.0 * float(data.qvel[3])
        data.qfrc_applied[4] += -250.0 * float(data.qpos[5]) - 30.0 * float(data.qvel[4])
    mujoco.mj_step(model, data)
    sim_steps += 1
    return sim_steps, control_steps, previous_action, target_joint_qpos


def _configure_viewer_camera(viewer_handle) -> None:
    viewer_handle.cam.lookat[:] = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    viewer_handle.cam.distance = 3.0
    viewer_handle.cam.elevation = -20.0
    viewer_handle.cam.azimuth = 135.0


if __name__ == "__main__":
    raise SystemExit(main())
