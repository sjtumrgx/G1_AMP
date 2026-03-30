from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyboardCommandLimits:
    lin_vel_x_min: float
    lin_vel_x_max: float
    ang_vel_z_min: float
    ang_vel_z_max: float


def resolve_keyboard_command_limits(env_cfg) -> KeyboardCommandLimits:
    base_velocity_cfg = _get_base_velocity_cfg(env_cfg)
    velocity_ranges = _get_value(base_velocity_cfg, "velocity_ranges")
    lin_vel_x_min = min(float(v["lin_vel_x"][0]) for v in velocity_ranges.values())
    lin_vel_x_max = max(float(v["lin_vel_x"][1]) for v in velocity_ranges.values())
    if _get_value(base_velocity_cfg, "only_positive_lin_vel_x"):
        lin_vel_x_min = max(0.0, lin_vel_x_min)
    ang_vel_z_min = min(float(v["ang_vel_z"][0]) for v in velocity_ranges.values())
    ang_vel_z_max = max(float(v["ang_vel_z"][1]) for v in velocity_ranges.values())
    return KeyboardCommandLimits(
        lin_vel_x_min=lin_vel_x_min,
        lin_vel_x_max=lin_vel_x_max,
        ang_vel_z_min=ang_vel_z_min,
        ang_vel_z_max=ang_vel_z_max,
    )


class ParkourKeyboardCommandController:
    def __init__(
        self,
        *,
        num_envs: int,
        history_length: int,
        device: torch.device | str,
        limits: KeyboardCommandLimits,
        linvel_step: float,
        angvel: float,
    ):
        self.num_envs = num_envs
        self.history_length = history_length
        self.device = device
        self.limits = limits
        self.linvel_step = linvel_step
        self.angvel = angvel
        self.command = torch.zeros(num_envs, 3, device=device)
        self.history = torch.zeros(num_envs, history_length, 3, device=device)

    def forward(self) -> None:
        self.command[:, 0] = torch.clamp(
            self.command[:, 0] + self.linvel_step,
            min=self.limits.lin_vel_x_min,
            max=self.limits.lin_vel_x_max,
        )

    def reset_yaw(self) -> None:
        self.command[:, 2] = 0.0

    def yaw_positive(self) -> None:
        self.command[:, 2] = self.limits.ang_vel_z_max if self.angvel > 0 else 0.0

    def yaw_negative(self) -> None:
        self.command[:, 2] = self.limits.ang_vel_z_min if self.angvel > 0 else 0.0

    def zero_all(self) -> None:
        self.command.zero_()

    def build_observation(self) -> torch.Tensor:
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = self.command
        return self.history.reshape(self.num_envs, -1)


def _get_base_velocity_cfg(env_cfg):
    if isinstance(env_cfg, dict):
        return env_cfg["commands"]["base_velocity"]
    return env_cfg.commands.base_velocity


def _get_value(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key)
