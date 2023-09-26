import torch
from pytorch_lightning import LightningModule


class classification_model(LightningModule):
    """ Simple linear layer (linear probe) 
    """
    def __init__(self, encoder, classifier,
            optimizer_name='adam',
            lr=0.001,
            freeze_encoder=True,
            **kwargs):
        super().__init__()
        self.name = 'classification_model'
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        X,Y = batch[0],batch[1]
        out = self(X)
        loss = self.loss(out,Y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X,Y = batch[0],batch[1]
        out = self(X)
        loss = self.loss(out,Y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        X,Y = batch[0],batch[1]
        out = self(X)
        loss = self.loss(out,Y)
        self.log("test_loss", loss)
    
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