from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable, Sequence


def select_onnx_runtime_providers(provider: str = "cpu") -> list[str]:
    """Resolve ONNX Runtime providers for Isaac Sim play.

    Defaulting to CPU avoids hangs/crashes when Isaac Sim, PyTorch, and ONNX Runtime all
    compete for the same CUDA context during interactive play/export.
    """
    provider = provider.lower()
    available = list(ort.get_available_providers())
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "Requested onnxruntime provider 'cuda' but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        ordered = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            ordered.append("CPUExecutionProvider")
        return ordered
    if provider == "cpu":
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        return available
    raise ValueError(f"Unsupported ONNX provider: {provider}")


def export_parkour_actor_critic_as_onnx(actor_critic, observations: torch.Tensor, model_dir: str) -> None:
    """Export ONNX via a CPU clone to avoid in-process Isaac Sim/CUDA exporter instability."""
    os.makedirs(model_dir, exist_ok=True)
    cpu_actor_critic = copy.deepcopy(actor_critic).cpu().eval()
    cpu_observations = observations.detach().to(device="cpu")
    print(
        "[INFO] Exporting ONNX policy with a CPU clone to avoid Isaac Sim/CUDA export hangs:"
        f" {model_dir}"
    )
    cpu_actor_critic.export_as_onnx(cpu_observations, model_dir)



def load_parkour_onnx_model(
    model_dir: str,
    get_subobs_func: Callable,
    depth_shape: tuple,
    proprio_slice: slice,
    providers: Sequence[str] | None = None,
    provider: str = "cpu",
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = list(providers) if providers is not None else select_onnx_runtime_providers(provider)
    encoder_path = os.path.join(model_dir, "0-depth_encoder.onnx")
    actor_path = os.path.join(model_dir, "actor.onnx")
    missing = [str(path) for path in (encoder_path, actor_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Expected exported ONNX files are missing. "
            f"Checked: {', '.join(missing)}"
        )
    print(f"[INFO] Loading ONNX policy from {model_dir} with providers={ort_providers}")
    encoder = ort.InferenceSession(encoder_path, providers=ort_providers)
    actor = ort.InferenceSession(actor_path, providers=ort_providers)
    actor_input_name = actor.get_inputs()[0].name

    def policy(obs: torch.Tensor) -> torch.Tensor:
        depth_image_input = get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = encoder.run(None, {encoder.get_inputs()[0].name: depth_image_input})[0]
        actor_input = np.concatenate(
            [
                obs.cpu().numpy()[:, proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        actor_output = actor.run(None, {actor_input_name: actor_input})[0]
        return torch.from_numpy(actor_output).to(obs.device)

    return policy
