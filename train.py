import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.loss_utils import ColorBalanceLoss, GradientStructureLoss
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(
        self,
        water_aware=False,
        frequency_refinement=False,
        local_contrast_refinement=False,
        lambda_color=0.0,
        lambda_structure=0.0,
        lr=2e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(
            decoder=True,
            water_aware=water_aware,
            frequency_refinement=frequency_refinement,
            local_contrast_refinement=local_contrast_refinement,
        )
        self.loss_fn = nn.L1Loss()
        self.color_loss = ColorBalanceLoss()
        self.structure_loss = GradientStructureLoss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        rec_loss = self.loss_fn(restored, clean_patch)
        color_loss = self.color_loss(restored)
        structure_loss = self.structure_loss(restored, clean_patch)
        loss = (
            rec_loss
            + self.hparams.lambda_color * color_loss
            + self.hparams.lambda_structure * structure_loss
        )
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_rec_loss", rec_loss)
        self.log("train_color_loss", color_loss)
        self.log("train_structure_loss", structure_loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel(
        water_aware=opt.water_aware,
        frequency_refinement=opt.frequency_refinement,
        local_contrast_refinement=opt.local_contrast_refinement,
        lambda_color=opt.lambda_color,
        lambda_structure=opt.lambda_structure,
        lr=opt.lr,
    )
    if opt.pretrain_ckpt is not None:
        ckpt = torch.load(opt.pretrain_ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Loaded pretrain checkpoint with "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys."
        )
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
