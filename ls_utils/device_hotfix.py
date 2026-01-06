import torch
import logging
logger = logging.getLogger(__name__)

ATTRS_BY_CLASS = {
    "RigidNodes": ("instances_fv", "point_ids", "instances_size"),
    "SMPLNodes": ("point_ids", "instances_fv", "instances_size",),
    "DeformableNodes": ("point_ids", "instances_fv", "instances_size",),
}

def move_node_tensors_to_device(trainer_obj):
    models = getattr(trainer_obj, "models", {})
    for _, module in models.items():
        cls = module.__class__.__name__
        attrs = ATTRS_BY_CLASS.get(cls)
        if not attrs:
            continue
        dev = getattr(module, "device", None)
        if dev is None:
            try:
                dev = next(module.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for attr in attrs:
            t = getattr(module, attr, None)
            if isinstance(t, torch.Tensor) and t.device != dev:
                setattr(module, attr, t.to(dev, non_blocking=True))
                logger.info(f"Moved {cls}.{attr} to {dev}")