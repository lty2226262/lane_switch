import sys
import torch
from .math_utils import make_lateral_offset_pose

def parse_lateral_offset_from_cli(argv) -> float:
    for arg in argv:
        if arg.startswith("render.render_novel.offset="):
            try:
                return float(arg.split("=", 1)[1])
            except Exception:
                pass
    return 0.0

def install_fixed_offset_trajectory(LATERAL: float):
    import utils.camera as cam
    _original_get_traj = cam.get_interp_novel_trajectories

    def _fixed_offset(dataset_type, scene_idx, per_cam_poses, traj_type, target_frames):
        ref_cam_id = 0 if 0 in per_cam_poses else sorted(per_cam_poses.keys())[0]
        poses = per_cam_poses[ref_cam_id]  # sequence of Tcw, len = num_frames
        T_list = []
        for Tcw in poses:
            Tcw_t = torch.as_tensor(Tcw, dtype=torch.float32)
            T_list.append(make_lateral_offset_pose(Tcw_t, LATERAL))
        Tw_shift = torch.stack(T_list, dim=0)  # [F,4,4]
        # If target_frames is smaller, truncate; if larger, pad last
        if target_frames is not None:
            if Tw_shift.shape[0] >= target_frames:
                Tw_shift = Tw_shift
            else:
                pad = Tw_shift[-1:].repeat(target_frames - Tw_shift.shape[0], 1, 1)
                Tw_shift = torch.cat([Tw_shift, pad], dim=0)
        return Tw_shift

    def get_interp_novel_trajectories(dataset_type, scene_idx, per_cam_poses, traj_type="front_center_interp", target_frames=100):
        if traj_type == "fixed_offset":
            return _fixed_offset(dataset_type, scene_idx, per_cam_poses, traj_type, target_frames)
        return _original_get_traj(dataset_type, scene_idx, per_cam_poses, traj_type, target_frames)

    cam.get_interp_novel_trajectories = get_interp_novel_trajectories

def install_render_traj_key_renamer(LATERAL: float):
    import datasets.driving_dataset as dd
    _original_get_novel_render_traj = dd.DrivingDataset.get_novel_render_traj

    def _rename_fixed_offset_keys(render_traj_dict):
        suffix = f"{float(LATERAL):+.2f}m"
        renamed = {}
        for k, v in render_traj_dict.items():
            new_k = f"fixed_offset_{suffix}" if k == "fixed_offset" else k
            renamed[new_k] = v
        return renamed

    def get_novel_render_traj(self, traj_types, target_frames=None):
        res = _original_get_novel_render_traj(self, traj_types, target_frames)
        if any(t == "fixed_offset" for t in traj_types):
            res = _rename_fixed_offset_keys(res)
        return res

    dd.DrivingDataset.get_novel_render_traj = get_novel_render_traj

def install_post_resume_hotfix():
    from .device_hotfix import move_node_tensors_to_device
    from models.trainers import MultiTrainer as Trainer

    original_resume = Trainer.resume_from_checkpoint

    def wrapped_resume(self, ckpt_path, load_only_model=False, *args, **kwargs):
        ret = original_resume(self, ckpt_path, load_only_model=load_only_model, *args, **kwargs)
        move_node_tensors_to_device(self)
        return ret

    Trainer.resume_from_checkpoint = wrapped_resume