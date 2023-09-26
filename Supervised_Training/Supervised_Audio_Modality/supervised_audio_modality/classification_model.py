import torch
import torch.nn as nn

class classification_model(nn.Module):
    """ Simple linear layer (linear probe) 
    """
    def __init__(self, encoder, classifier,
            optimizer_name='adam',
            lr=0.001,
            **kwargs):
        super().__init__()
        self.name = 'classification_model'
        self.encoder = encoder
        self.classifier = classifier

        self.optimizer_name = optimizer_name
        self.lr = lr

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }