"""isomerge: isotropy and LoRA composability on JEPA backbones.

Subpackages:
  data     -- the 8-task vision-merging suite + synthetic toy tasks
  models   -- unified encoder wrappers (V-JEPA 2, I-JEPA, MAE, DINOv2, LeJEPA ViT-S)
  adapt    -- LoRA injection, SIGReg loss, fine-tuning trainer
  merging  -- uniform / task-arithmetic / TIES / DARE-TIES / rd_encoder
  metrics  -- Arm C library: isotropy, task-vector geometry, functional interference
  eval     -- P1 linear probe, P2 head reuse, retention + statistics
  pretrain -- LeJEPA-style ViT-S pretraining with the SIGReg lambda dial (Arm B)
"""

__version__ = "0.1.0"
