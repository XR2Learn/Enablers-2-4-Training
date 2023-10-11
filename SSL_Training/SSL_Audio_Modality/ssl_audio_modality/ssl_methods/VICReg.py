import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

class VICReg(LightningModule):
    """
    Implementation of VicReg adapted from https://github.com/facebookresearch/vicreg/
    """
    def __init__(self, encoder, ssl_batch_size=128, sim_coeff = 25, std_coeff = 25, cov_coeff = 1, optimizer_name_ssl='adam', lr=0.005, **kwargs):

        super().__init__()

        self.encoder = encoder
        # TODO: make projection heads customizable (number of neurons)
        self.projection = nn.Sequential(
                nn.Linear(self.encoder.out_size, 512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
        )

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr

        self.embedding_size = self.encoder.out_size
        self.ssl_batch_size = ssl_batch_size

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def on_fit_start(self):
        self.encoder.to(self.device)

    def _process_batch(self, batch):
        aug1, aug2 = batch[0].float(), batch[-1].float()
        return aug1,aug2

    def _compute_vicreg_loss(self, x, y, partition):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"{partition}_loss", loss)

        return loss
    
    def forward(self,x,y):
        x = self.projection(nn.Flatten()(self.encoder(x)))
        y = self.projection(nn.Flatten()(self.encoder(y)))
        return x, y

    def training_step(self, batch, batch_idx):
        aug,x = self._process_batch(batch)
        x, y = self(x,aug)
        return self._compute_vicreg_loss(x, y, 'train')

    def validation_step(self, batch, batch_idx):
        aug,x = self._process_batch(batch)
        x, y = self(x,aug)
        return self._compute_vicreg_loss(x, y, 'val')
    
    def test_step(self, batch, batch_idx):
        aug,x = self._process_batch(batch)
        x, y = self(x,aug)
        return self._compute_vicreg_loss(x, y, 'test')

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=7)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'train_loss'
                }
            }

        #elif self.optimizer_name_ssl.lower() == 'lars':
        #    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #    optimizer = LARC(optimizer)
        #    return {
        #        "optimizer": optimizer
        #    }

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()