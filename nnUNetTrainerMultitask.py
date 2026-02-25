import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score, accuracy_score
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context


class CrossAttentionPooling(nn.Module):
    """
    Replaces GAP + MLP with learnable query tokens that cross-attend to the
    encoder bottleneck features, preserving spatial discriminative information.
    """

    def __init__(self, embed_dim: int, num_queries: int = 3,
                 num_heads: int = 8, num_classes: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_queries * embed_dim),
            nn.Linear(num_queries * embed_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, C, D, H, W) -> flatten spatial -> (B, DHW, C)
        B = x.shape[0]
        x = x.flatten(2).permute(0, 2, 1)                # (B, S, C)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, Q, C)
        q = self.norm(self.cross_attn(q, x, x)[0])        # (B, Q, C)
        return self.head(q)                                # (B, num_classes)


class MultiTaskWrapper(nn.Module):
    """
    Wraps a segmentation U-Net (PlainConvUNet or ResidualEncoderUNet) and adds
    a lightweight classification head that branches off the encoder bottleneck.

    Architecture rationale:
    - The encoder already learns rich spatial features; a shallow classifier
      on top of Global-Average-Pooled bottleneck features is sufficient and
      avoids overfitting on the small 3-class classification task (252 samples).
    - Two FC layers with dropout are used instead of one to give the head
      enough capacity to separate the 3 subtypes while still being lightweight.
    """

    def __init__(self, base_network: nn.Module, num_classes_cls: int = 3,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.base_network = base_network

        # Both PlainConvEncoder and ResidualEncoder store output_channels
        bottleneck_dim = base_network.encoder.output_channels[-1]

        self.cls_head = CrossAttentionPooling(
            embed_dim=bottleneck_dim, num_queries=3, num_heads=8,
            num_classes=num_classes_cls, dropout=dropout_rate,
        )

        # When False, forward() returns only segmentation output (for inference)
        self._do_cls = True

    # ---- delegate encoder / decoder so set_deep_supervision_enabled works ----
    @property
    def decoder(self):
        return self.base_network.decoder

    @property
    def encoder(self):
        return self.base_network.encoder

    def forward(self, x):
        skips = self.base_network.encoder(x)
        seg_output = self.base_network.decoder(skips)

        if self._do_cls:
            bottleneck = skips[-1]                # (B, C, D', H', W')
            cls_output = self.cls_head(bottleneck)  # (B, num_classes_cls)
            return seg_output, cls_output

        return seg_output

    def compute_conv_feature_map_size(self, input_size):
        return self.base_network.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        """Delegate weight init to the base network's own initializer."""
        pass


# ---------------------------------------------------------------------------
# Multi-task nnUNet trainer
# ---------------------------------------------------------------------------
class nnUNetTrainerMultiTask(nnUNetTrainer):
    """
    Extends the default nnUNet trainer to jointly train:
      1. Segmentation (pancreas + lesion) – standard Dice + CE loss
      2. Classification (3 lesion subtypes) – weighted Cross-Entropy

    Classification labels are derived at runtime from the case identifiers
    (e.g. "quiz_0_041" → subtype 0), so no changes to dataset.json are needed.

    WandB is used to track all losses and metrics.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_classes_cls = 3
        # Weight for classification loss relative to segmentation loss.
        # Both losses are roughly in the same magnitude range, so 1.0 is a
        # reasonable starting point.
        self.cls_loss_weight = 0.1

        self._wandb_initialized = False

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the base segmentation U-Net, then wrap it with a cls head."""
        base_network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return MultiTaskWrapper(base_network, num_classes_cls=3)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self):
        super().initialize()

        # Class-weighted CrossEntropy for imbalanced subtypes
        # Training distribution: subtype0=62, subtype1=106, subtype2=84
        counts = np.array([62.0, 106.0, 84.0])
        inv_freq = 1.0 / counts
        weights = inv_freq / inv_freq.sum() * len(counts)
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
        )

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    def _init_wandb(self):
        if not self._wandb_initialized and self.local_rank == 0:
            wandb.init(
                project="pancreas-multitask",
                name=f"{self.__class__.__name__}_fold{self.fold}",
                config={
                    "architecture": self.configuration_manager.network_arch_class_name,
                    "patch_size": list(self.configuration_manager.patch_size),
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "initial_lr": self.initial_lr,
                    "cls_loss_weight": self.cls_loss_weight,
                },
            )
            self._wandb_initialized = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subtype(key: str) -> int:
        """
        Derive subtype label from case identifier.
        Format: quiz_{subtype}_{case_id}  →  subtype is parts[1].
        """
        return int(key.split("_")[1])

    def _get_network_mod(self):
        """Unwrap DDP / torch.compile to reach the MultiTaskWrapper."""
        mod = self.network
        if self.is_ddp:
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()
        self._init_wandb()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cls_labels = torch.tensor(
            [self._parse_subtype(k) for k in keys],
            dtype=torch.long, device=self.device,
        )

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.cls_loss_fn(cls_output, cls_labels)
            total_loss = seg_loss + self.cls_loss_weight * cls_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'cls_loss': cls_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: list):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            import torch.distributed as dist
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        seg_loss_here = np.mean(outputs['seg_loss'])
        cls_loss_here = np.mean(outputs['cls_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        if self._wandb_initialized:
            wandb.log({
                'train/total_loss': float(loss_here),
                'train/seg_loss': float(seg_loss_here),
                'train/cls_loss': float(cls_loss_here),
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch': self.current_epoch,
            })

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cls_labels = torch.tensor(
            [self._parse_subtype(k) for k in keys],
            dtype=torch.long, device=self.device,
        )

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.cls_loss_fn(cls_output, cls_labels)
            total_loss = seg_loss + self.cls_loss_weight * cls_loss

        # --- Segmentation dice metrics (same logic as base class) ---
        if self.enable_deep_supervision:
            seg_out = seg_output[0]
            target_seg = target[0]
        else:
            seg_out = seg_output
            target_seg = target

        axes = [0] + list(range(2, seg_out.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_out) > 0.5).long()
        else:
            output_seg = seg_out.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                seg_out.shape, device=seg_out.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_seg != self.label_manager.ignore_label).float()
                target_seg[target_seg == self.label_manager.ignore_label] = 0
            else:
                if target_seg.dtype == torch.bool:
                    mask = ~target_seg[:, -1:]
                else:
                    mask = 1 - target_seg[:, -1:]
                target_seg = target_seg[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target_seg, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # --- Classification predictions ---
        cls_preds = cls_output.argmax(dim=1).detach().cpu().numpy()
        cls_targets = cls_labels.detach().cpu().numpy()

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'cls_preds': cls_preds,
            'cls_targets': cls_targets,
        }

    def on_validation_epoch_end(self, val_outputs: list):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)
            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)
            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        # Segmentation dice
        global_dc_per_class = [
            2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class,
                        self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # Classification metrics
        all_preds = np.concatenate(outputs_collated['cls_preds'])
        all_targets = np.concatenate(outputs_collated['cls_targets'])
        macro_f1 = f1_score(all_targets, all_preds, average='macro',
                            zero_division=0)
        cls_accuracy = accuracy_score(all_targets, all_preds)

        self.print_to_log_file(
            f"Val seg -- mean Dice: {mean_fg_dice:.4f}, "
            f"per-class: {[round(d, 4) for d in global_dc_per_class]}")
        self.print_to_log_file(
            f"Val cls -- macro-F1: {macro_f1:.4f}, accuracy: {cls_accuracy:.4f}")

        if self._wandb_initialized:
            log_dict = {
                'val/total_loss': float(loss_here),
                'val/mean_fg_dice': float(mean_fg_dice),
                'val/cls_macro_f1': float(macro_f1),
                'val/cls_accuracy': float(cls_accuracy),
                'epoch': self.current_epoch,
            }
            for idx, d in enumerate(global_dc_per_class):
                log_dict[f'val/dice_class_{idx}'] = float(d)
            wandb.log(log_dict)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        """Works for both base UNet and MultiTaskWrapper."""
        mod = self._get_network_mod()
        mod.decoder.deep_supervision = enabled

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Temporarily disable classification head so the predictor receives
        only segmentation logits (same shape the base class expects).
        """
        mod = self._get_network_mod()
        mod._do_cls = False
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            mod._do_cls = True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def on_train_end(self):
        super().on_train_end()
        if self._wandb_initialized:
            wandb.finish()
