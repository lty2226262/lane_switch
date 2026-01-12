<h1 align="center">🛣️ <em>Lane-Switch Gaussian Splatting</em>: Urban Scene View Extrapolation via Selective Diffusion Refinement</h1>

<div align="center">
    <p>
        <!-- Fill authors as needed -->
        <a href="https://scholar.google.com/citations?user=NAt3vgcAAAAJ&hl=en">Tianyu Liu</a><sup>1</sup>
        <a href="https://scholar.google.com/citations?user=yRAHVcgAAAAJ&hl=en">Yuan Liu</a><sup>1†</sup>
        <a href="[AUTHOR_LINK_TBD]">Bin Ma</a><sup>2</sup>
        <a href="https://scholar.google.com/citations?user=liaSLT8AAAAJ&hl=en">Kunming Luo</a><sup>1</sup>
        <a href="https://scholar.google.com/citations?user=XhyKVFMAAAAJ&hl=en">Ping Tan</a><sup>1†</sup>&nbsp;&nbsp;
    </p>
    <p>
        <sup>1</sup>The Hong Kong University of Science and Technology &nbsp;&nbsp;&nbsp;
        <sup>2</sup>Meta
    </p>
    <p>
        <sup>†</sup> Corresponding Author
    </p>
</div>

<p align="center">
    <a href="[PAPER_LINK_TBD]" target="_blank">
    <img src="https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="[PROJECT_PAGE_LINK_TBD]" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page">
    </a>
</p>

<div align="center">
    <a href="[PROJECT_PAGE_LINK_TBD]">
        <img src="assets/main.gif" width="50%">
    </a>
    <p>
        <i>Lane-Switch Gaussian Splatting refines extrapolated view synthesis for lane-switch scenarios, mitigating unwanted over-correction in StableDiffusion's generated results and requires no per-scene fine-tuning.</i>
    </p>
</div>


## 📣 Updates
* **[January 12, 2026]** ✨ Integrated GaussianRefiner with **Difix3D+** for faster inference and with **Drivestudio** for multi-dataloader and advanced 4DGS reconstruction fitting.
* **[Legacy]** 🔧 Released baseline code and the UnClip-based GaussianRefiner's that is introduced in the paper, check the diffusion's training code `gs_refiner_training/legacy`.


## ✨ Overview
Lane-Switch Gaussian Splatting is a plugin for existing autonomous driving simulators designed specifically for lateral view extrapolation in lane-switch scenarios! Its core strength is progressive refinement: the GaussianRefiner fills in artifacts and missing content, while the Adaptive Refinement Arbiter enables selective optimization (preserving geometric consistency in high-quality regions without over-correction). No scene-specific fine-tuning is needed, it maintains real-time rendering at 112 FPS, and boasts strong generalization to unseen datasets.




## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/lty2226262/lane_switch.git
cd lane_switch

# Create the environment
conda create -n lane_switch python=3.9 -y
conda activate lane_switch
pip install --no-build-isolation -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd drivestudio/third_party/smplx/
pip install .
cd ../../../
```

### 2. Download Preprocessed Reconstruction Data (Optional)
```bash

huggingface-cli download \
  lty2226262/lane_switch_gs_dataset smpl_models.zip --repo-type dataset --local-dir .

unzip smpl_models.zip

huggingface-cli download \
  lty2226262/lane_switch_gs_dataset waymo_023.zip --repo-type dataset --local-dir .

unzip waymo_023.zip
```


## 🧪 Usage

### Evaluate Extrapolated View Synthesis Before Refinement
```bash
python lane_switch_eval.py \
  --resume_from data/waymo/processed/training/023/ckpts/checkpoint_init.pth \
  render.render_novel.traj_types=[fixed_offset] \
  render.render_novel.offset=-3 \
  render.render_full=false \
  render.render_test=false
```

### Refinement
```bash
python lane_switch_train.py \
  --config_file configs/omnire.yaml \
  --output_root result_finetune \
  --project waymo \
  --run_name scene23offset3m \
  --resume_from data/waymo/processed/training/023/ckpts/checkpoint_init.pth \
  dataset=waymo/1cams_finetune \
  data.scene_idx=23 \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  trainer.optim.num_iters=300 \
  render.render_novel.offset=-3
```

### Evaluate Extrapolated View Synthesis After Refinement
```bash
python lane_switch_eval.py \
  --resume_from result_finetune/waymo/scene23offset-3m/checkpoint_final.pth \
  render.render_novel.traj_types=[fixed_offset] \
  render.render_novel.offset=-3 \
  render.render_full=false \
  render.render_test=false
```


## 🛠️ Notes & Tips

<details>
<summary><b>Customize your driving scenario</b></summary>

Reconstruct your own scene with Drivestudio, then create an “init” checkpoint by resetting the training `step` to `0`. The generated `checkpoint_init.pth` will be saved in the **same directory** as the input checkpoint.

```bash
python - <<'PY'
import argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", help="Path to an existing checkpoint (e.g., checkpoint_final.pth)")
args = ap.parse_args()

src = Path(args.ckpt).expanduser().resolve()
dst = src.with_name("checkpoint_init.pth")

ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
ckpt["step"] = 0
torch.save(ckpt, str(dst))

print("Saved:", dst)
PY <your_checkpoint_path>
```

</details>

<details>
<summary><b>Troubleshooting: large black holes / black regions in renders, or blurred refinements</b></summary>

Because the pipeline follows an iterative **reconstruct → refine → reconstruct → refine** pattern, the reconstruction quality strongly affects the refinement quality. If you see large missing regions (black holes) or noticeably blurred refinement results, try the following:

1. **Inspect refiner debug outputs**
   - In [`ls_utils/train_utils.py#L5`](ls_utils/train_utils.py#L5), set `DEBUG_OUTPUT = True`.
   - Then check images under `debug_refiner_outputs/` to see whether the refiner inputs/conditions are misaligned.

2. **Common root causes to verify**
   - **Incorrect bounding boxes / cropping** (e.g., the region of interest is off).
   - **Ego-vehicle pose errors** (pose estimation drift can cause severe misalignment).
   - **Camera calibration issues** (intrinsics/extrinsics mismatch leading to blur/ghosting).

3. **If the original rendered input already has large missing regions**
   - Try using a **smaller lateral offset** first.
   - Or **fine-tune a customized GaussianRefiner** for your customized scenario. See
     [`gs_refiner_training/legacy/README.md`](gs_refiner_training/legacy/README.md)
     (includes Difix3D/Z-Image fine-tuning references).

</details>

## 🙏 Acknowledgements
We gratefully acknowledge the following projects and libraries:
- [VEGS](https://github.com/facebookresearch/pytorch3d)
- [IP-Adapter](https://github.com/NVlabs/nvdiffrast)
- [Drivestudio](https://github.com/huggingface/diffusers)
- [Difix](https://github.com/nerfstudio-project/gsplat)


## 📜 Citation
If you find this work useful, please consider citing (TBD):
```bibtex
@article{laneswitch2026,
  title={Lane-Switch Gaussian Splatting: Urban Scene View Extrapolation via Selective Diffusion Refinement},
  author={Liu, Tianyu and Liu, Yuan and Ma, Bin and Luo, Kunming and Tan, Ping},
  journal={arXiv preprint},
  year={2026}
}
```


## 📄 License
For academic use, this project is licensed under the 2-clause BSD License. See the [LICENSE](./LICENSE) file for details. For commercial use, please contact the authors.
