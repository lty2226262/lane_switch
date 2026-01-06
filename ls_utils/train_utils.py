import torch
from typing import Optional
import numpy as np
from PIL import Image
from pytorch_msssim import SSIM  # add import
from .gsrefiner import GSRefinerPipeline
from .math_utils import make_lateral_offset_pose
from datasets.base.pixel_source import get_rays


_REFINE_MODEL_SINGLETON: Optional[torch.nn.Module] = None

class _GSRefinerWrapper(torch.nn.Module):
    """
    Thin wrapper to expose a .forward(gt_rgb, shifted_rgb) API for GSRefinerPipeline/Difix.
    gt_rgb, shifted_rgb: float tensors in [0,1], shape [B,H,W,3] or [H,W,3].
    """
    def __init__(self, pretrained_id: str = "nvidia/difix_ref", device: torch.device = torch.device("cuda")):
        super().__init__()
        self.pipe = GSRefinerPipeline.from_pretrained(pretrained_id, trust_remote_code=True)
        # Use fp16 to reduce VRAM
        self.pipe.to(device, torch.float16)
        # Enable memory-saving features (diffusers)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            # Optional: more aggressive VRAM saving, slower
            # self.pipe.enable_model_cpu_offload()
            pass
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        # Attach SSIM for reuse (expects NCHW, data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)

    @staticmethod
    def _to_pil(img: torch.Tensor):
        # img: [..., H, W, 3], [0,1]
        if img.ndim == 4:
            img = img[0]  # take first item if batch
        arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    @staticmethod
    def _to_tensor(img_pil):
        arr = np.array(img_pil).astype("float32") / 255.0
        t = torch.from_numpy(arr)  # [H,W,3] on CPU
        return t

    def forward(self, gt_rgb: torch.Tensor, shifted_rgb: torch.Tensor) -> torch.Tensor:
        # Keep refiner inference-only; do not store graphs
        with torch.inference_mode():
            gt_pil = self._to_pil(gt_rgb)
            shifted_pil = self._to_pil(shifted_rgb)
            out = self.pipe(
                "remove degradation",
                image=shifted_pil,
                ref_image=gt_pil,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
            )
            repaired_pil = out.images[0]
            repaired_t = self._to_tensor(repaired_pil).to(gt_rgb.device)
        # cleanup locals to help GC
        del out, gt_pil, shifted_pil, repaired_pil
        if gt_rgb.ndim == 4:
            repaired_t = repaired_t.unsqueeze(0)
        return repaired_t

def get_refiner(cfg, device: torch.device, reload: bool = False) -> torch.nn.Module:
    global _REFINE_MODEL_SINGLETON
    if _REFINE_MODEL_SINGLETON is not None and not reload:
        return _REFINE_MODEL_SINGLETON
    refine_cfg = getattr(cfg, "refiner", None)
    pretrained_id = getattr(refine_cfg, "pretrained_id", "nvidia/difix_ref") if refine_cfg else "nvidia/difix_ref"
    model = _GSRefinerWrapper(pretrained_id=pretrained_id, device=device)
    _REFINE_MODEL_SINGLETON = model
    return _REFINE_MODEL_SINGLETON

def release_refiner():
    global _REFINE_MODEL_SINGLETON
    _REFINE_MODEL_SINGLETON = None

def _resize_cam_and_scale_intrinsics(cam_infos_shifted: dict, target_w=960, target_h=640):
    H_ori = cam_infos_shifted.get("height", None)
    W_ori = cam_infos_shifted.get("width", None)
    K = cam_infos_shifted.get("intrinsics", None)
    if H_ori is None or W_ori is None or K is None:
        return cam_infos_shifted

    # Allow scalar tensors
    W = float(W_ori.item())
    H = float(H_ori.item())

    aspect = float(W) / float(H)
    target_aspect = float(target_w) / float(target_h)
    if aspect < target_aspect:
        scale = target_w / float(W)
    else:
        scale = target_h / float(H)
    
    new_W = int(W * scale)
    new_H = int(H * scale)

    cam_infos_shifted["width"] = int(W_ori * scale)
    cam_infos_shifted["height"] = int(H_ori * scale)

    K = K.clone()
    K[..., 0, 0] *= scale  # fx
    K[..., 1, 1] *= scale  # fy
    K[..., 0, 2] *= scale  # cx
    K[..., 1, 2] *= scale  # cy

    cam_infos_shifted["intrinsics"] = K

    x0 = torch.randint(0, max(1, new_W - target_w + 1), (1,)).item() if new_W > target_w else 0
    y0 = torch.randint(0, max(1, new_H - target_h + 1), (1,)).item() if new_H > target_h else 0
    x1 = min(new_W, x0 + target_w)
    y1 = min(new_H, y0 + target_h)
    return cam_infos_shifted, scale, (x0, y0, x1, y1)

def _render_with_lateral_offset(trainer, image_infos: dict, cam_infos: dict, lateral_m: float) -> dict:
    """Render with a lateral camera translation (keeps gradients and train mode)."""
    if lateral_m == 0.0:
        return {}
    cam_infos_shifted = {k: v for k, v in cam_infos.items()}
    image_infos_shifted = {k: v for k, v in image_infos.items()}
    Tcw = cam_infos_shifted.get("camera_to_world", None)  # adjust key if your data uses "Tcw"
    if Tcw is None:
        return {}
    cam_infos_shifted["camera_to_world"] = make_lateral_offset_pose(Tcw, lateral_m)
    cam_infos_shifted, scale, (x0, y0, x1, y1) = _resize_cam_and_scale_intrinsics(cam_infos_shifted, target_w=960, target_h=640)

    H_new = cam_infos_shifted["height"]
    W_new = cam_infos_shifted["width"]
    x, y = torch.meshgrid(torch.arange(W_new), torch.arange(H_new), indexing='xy')
    x = x.to(cam_infos_shifted["intrinsics"].device)
    y = y.to(cam_infos_shifted["intrinsics"].device)
    c2w = cam_infos_shifted["camera_to_world"]
    intrinsics = cam_infos_shifted["intrinsics"]

    origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, intrinsics)
    image_infos_shifted["origins"] = origins.reshape(H_new, W_new, 3)
    image_infos_shifted["viewdirs"] = viewdirs.reshape(H_new, W_new, 3)
    image_infos_shifted["direction_norm"] = direction_norm.reshape(H_new, W_new, 1)

    saved_affine_model = trainer.models.pop("Affine", None)
    outputs = trainer(image_infos_shifted, cam_infos_shifted, novel_view=True)
    trainer.models["Affine"] = saved_affine_model

    return outputs, scale, (x0, y0, x1, y1)

def add_lateral_refine_loss(
    trainer,
    cfg,
    image_infos: dict,
    cam_infos: dict,
    refiner: Optional[torch.nn.Module],
    lateral_m: float,
    weight: float = 1.0,
) -> tuple[str, torch.Tensor]:
    """
    Render with lateral offset -> refine (no grad) -> 0.8*L2 + 0.2*(1-SSIM).
    Gradients flow to 'shifted' (renderer).
    """
    device = getattr(trainer, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if weight == 0.0 or lateral_m == 0.0 or "pixels" not in image_infos:
        return "lateral_refine_loss", torch.tensor(0.0, device=device)

    gt = image_infos["pixels"]  # [..., H, W, 3], [0,1]
    outputs_shifted, scale, (x0, y0, x1, y1) = _render_with_lateral_offset(trainer, image_infos, cam_infos, lateral_m)
    shifted = outputs_shifted.get("rgb", None)
    if scale != 1.0:
        H_gt = gt.shape[0]
        W_gt = gt.shape[1]
        new_H = int(H_gt * scale)
        new_W = int(W_gt * scale)
        gt = torch.nn.functional.interpolate(
            gt.permute(2, 0, 1).unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    gt = gt[..., y0:y1, x0:x1, :]
    shifted = shifted[..., y0:y1, x0:x1, :]
    
    # Align batch if needed
    if shifted.shape != gt.shape:
        b = min(gt.shape[0], shifted.shape[0]) if gt.ndim == 4 else 1
        gt = gt[:b]
        shifted = shifted[:b]

    model = refiner if refiner is not None else get_refiner(cfg, device=device)
    # Refiner is inference-only; do not track gradients inside it
    with torch.inference_mode():
        refined = model(gt, shifted).clamp(0.0, 1.0)

    # 0.8 * L2 + 0.2 * (1 - SSIM), SSIM computed via model.ssim (NCHW)
    l2 = torch.mean((refined - shifted) ** 2)
    if refined.ndim == 3:
        refined_nchw = refined.permute(2, 0, 1)[None, ...]      # [1,3,H,W]
        shifted_nchw = shifted.permute(2, 0, 1)[None, ...]
    else:
        refined_nchw = refined.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        shifted_nchw = shifted.permute(0, 3, 1, 2).contiguous()
    refined_nchw = refined_nchw.clone()           # no grads needed
    shifted_nchw = shifted_nchw.clone() 
    
    ssim_val = model.ssim(refined_nchw, shifted_nchw)  # differentiable wrt shifted_nchw
    loss = 0.8 * l2 + 0.2 * (1.0 - ssim_val)

    # cleanup
    del outputs_shifted, shifted, refined, refined_nchw, shifted_nchw
    return "lateral_refine_loss", loss * float(weight)