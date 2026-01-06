import torch

def make_lateral_offset_pose(Tcw: torch.Tensor, lateral_m: float) -> torch.Tensor:
    # Tcw: [4,4] camera-to-world; camera +X is right, left(+L) => dx = -L
    R, t = Tcw[:3, :3], Tcw[:3, 3]
    dx, dy, dz = -float(lateral_m), 0.0, 0.0
    offset_world = R @ torch.tensor([dx, dy, dz], dtype=Tcw.dtype, device=Tcw.device)
    Tw_shift = Tcw.clone()
    Tw_shift[:3, 3] = t + offset_world
    return Tw_shift