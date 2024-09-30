from typing import Union

import torch
from pytorch_lightning import LightningModule


class SupervisedModel(LightningModule):
    """ LightningModule that takes various encoders and classifiers as inputs and performs supervised training
    """
    def __init__(
            self,
            encoder: Union[torch.nn.Module, LightningModule],
            classifier: Union[torch.nn.Module, LightningModule],
            optimizer_name: str = 'adam',
            lr: float = 0.001,
            freeze_encoder: bool = True,
            **kwargs
    ):
        super().__init__()
        self.name = 'supervised_model'
        self.encoder = encoder
        self.flatten = torch.nn.Flatten()
        self.classifier = classifier

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("succesfully froze the parameters for the encoder")
        else:
            print("NO paramaters frozen")

        ignore_list = []
        # It is expected that if components are instances of LightningModules, they trigger their own
        # self.save_hyperparameters() when initialized
        if isinstance(encoder, LightningModule):
            ignore_list.append('encoder')
        if isinstance(classifier, LightningModule):
            ignore_list.append('classifier')
        self.save_hyperparameters(ignore=ignore_list)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        X, Y = batch[0], batch[1]
        out = self(X)
        loss = self.loss(out, Y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val_")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test_")

    def _shared_eval(self, batch, batch_idx, prefix="val_"):
        X, Y = batch[0], batch[1]
        out = self(X)
        loss = self.loss(out, Y)
        self.log(f"{prefix}loss", loss)
        preds = torch.argmax(out, dim=1)
        return {f"{prefix}loss": loss, "preds": preds}

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'train_loss'
                }
            }
