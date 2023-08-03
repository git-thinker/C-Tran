from typing import *
from typing import List, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

import torch
import model
import util
import torchmetrics
import pytorch_lightning as pl

class ModelWrapper(pl.LightningModule):
    def __init__(
        self, 
        backbone_name: str,
        num_classes: int,
        embedding_dim: int = 2048,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        learning_rate: float = 1e-4,
        mask_rate=0.5,
        *args: Any, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model.CTran(
            backbone_name=backbone_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
        )
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.metrics = util.MetricsCentre()
    
    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        mask = torch.bernoulli(torch.full(y.shape, self.hparams.mask_rate)).to(y.device)
        y_hat = self.model(x, y, mask)
        # y_hat = self.model(x, y, torch.zeros_like(y).to(y.device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
        loss = (loss * mask).mean()
        # loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True,)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999))
        return optimizer 

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x, y, torch.zeros_like(y).to(y.device))
        y_hat = torch.sigmoid(y_hat)
        self.metrics.log(y, y_hat)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        report = self.metrics.evaluate()
        for k, v in report.items():
            self.log(k, v, on_epoch=True, prog_bar=True if '1' in k else False)
        self.metrics.clear()
        
