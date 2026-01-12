# GaussianRefiner Training Guide (Legacy)

This guide explains how to set up the training dataset, configure dependencies, and run training for the cross-attention repair model on single- or multi-GPU setups. 

💡Note: This legacy refiner is intended to investigate various cross-view feature transfer methods. For faster inference, we recommend fine-tuning your own GaussianRefiner based on Difix3D or Z-Image.

- Difix3D: https://github.com/nv-tlabs/Difix3D?tab=readme-ov-file#multipe-gpus
- Z-Image: https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md

## 1) Data layout
The loader expects `example_kitti.json` in the repo root. It contains three parallel lists: `render` (input), `gt` (target), and `cond` (conditioning image). Paths are relative to the repo root. Example (excerpt from [example_kitti.json](example_kitti.json)):
```json
{
  "render": [
    "render_files/output_both_cams_0_6623_6851_left/.../render_rgb/27500/0000006623.png"
  ],
  "gt": [
    "KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect/0000006623.png"
  ],
  "cond": [
    "KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000006623.png"
  ]
}
```
Key points:
- Lists must be the same length and aligned by index.
- Validation split: last 10 samples are held out internally; by default the dataloader reuses the train loader for validation logging.

## 2) Running training
The entrypoint is [train_repair_cross_attn_version.py](train_repair_cross_attn_version.py). Important args:
- `--pretrained_model_name_or_path` (required): Stable Diffusion 2.1 UNCLIP checkpoint, e.g. `stabilityai/stable-diffusion-2-1-unclip`.
- `--output_dir`: where checkpoints/logs go (default `sd21-unclip-model-finetuned-together`).
- `--train_batch_size`, `--gradient_accumulation_steps`, `--num_train_epochs` or `--max_train_steps`.
- `--mixed_precision {fp16,bf16}` for faster training.
- `--checkpointing_steps` and `--checkpoints_total_limit` to control checkpoint frequency/retention.

### Single GPU
```bash
accelerate launch --num_processes 1 --mixed_precision fp16 \
  train_repair_cross_attn_version.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
  --output_dir outputs/run_single \
  --train_batch_size 1 \
  --gradient_accumulation_steps 20 \
  --num_train_epochs 1 \
  --validation_epochs 1
```
Adjust batch/epochs as memory allows.

### Multi-GPU (single node)
```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
  train_repair_cross_attn_version.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
  --output_dir outputs/run_multi \
  --train_batch_size 1 \
  --gradient_accumulation_steps 10
```
- `--num_processes` = number of GPUs. For multi-node, add `--num_machines`, `--machine_rank`, and `--main_process_ip/port` per Accelerate docs.
- Learning rate scales with world size if `--scale_lr` is set.
