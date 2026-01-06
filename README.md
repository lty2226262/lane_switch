```shell
# Clone the repository with submodules
git clone --recursive https://github.com/lty2226262/lane_switch.git
cd lane_switch

# Create the environment
conda create -n lane_switch python=3.9 -y
conda activate lane_switch
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd drivestudio/third_party/smplx/
pip install .
cd ../../../
```


```
wget spml

unzip spml

（记住要写改ckpts的脚本，还有config.yaml文件的脚本

wget https://huggingface.co/datasets/lty2226262/lane_switch_gs_dataset/resolve/main/waymo_scene023.zip

unzip waymo_scene023.zip
```


Usage
```
CUDA_VISIBLE_DEVICES=2 python lane_switch_eval.py \
  --resume_from data/waymo/processed/training/023/ckpts/checkpoint_init.pth \
  render.render_novel.traj_types=[fixed_offset] \
  render.render_novel.offset=-3 \
  render.render_full=false \
  render.render_test=false


```

Finetune
```
CUDA_VISIBLE_DEVICES=1 python lane_switch_train.py \
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
    render.render_novel.offset=3
```

CUDA_VISIBLE_DEVICES=2 python lane_switch_eval.py \
  --resume_from result_finetune/waymo/scene23offset-3m/checkpoint_00250.pth \
  render.render_novel.traj_types=[fixed_offset] \
  render.render_novel.offset=-3 \
  render.render_full=false \
  render.render_test=false
