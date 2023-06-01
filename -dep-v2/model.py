from typing import Any, Callable
from pydantic import BaseModel, Field

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from torch.utils.data import DataLoader, Dataset, random_split

class Modelcfg(BaseModel):
        
	n_l: 			int     = Field(3, description= 'number of fermi block layers')

class CustomDataset(Dataset):
	"""Custom dataset class."""

	def __init__(self, data_path: str):
		self.data_path = data_path
		# Load your data from data_path

	def __len__(self) -> int:
		# Return the length of the dataset
		pass

	def __getitem__(self, idx: int) -> tuple:
		# Return a single data sample
		pass


class CustomTrainer(pl.LightningModule):
    """Custom PyTorch Lightning model class."""
    acc:        torch.Tensor                 = torch.tensor([0.5])
    deltar:     torch.Tensor                 = torch.tensor([0.02])
    sample_data:dict[str, torch.Tensor]|None = None

    def __init__(self, 
            c, 
            model: nn.Module, 
            loss_fn: nn.Module,
            sample: Callable,
        ):
        super().__init__()

        self.c = c
        self.model = model
        self.loss_fn = loss_fn
        self._sample = sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        return self.model(x)
    
    def sample(self, data: torch.Tensor= None, n_step: int= 1) -> torch.Tensor:
    	"""Sample from the neural network."""
    	deltar = self.sample_data.get('deltar', 0.02 * torch.ones_like(deltar))
    	acc = self.sample_data.get('acc', 0.5 * torch.ones_like(deltar))
    	data = data or self.sample_data.get('data')
    	v = dict(data= data, deltar= deltar, acc= acc)
    	for _ in range(n_step):
    		v = self.sample(**v)
    	self.sample_data = v.copy()
    	return v

    def training_step(self) -> torch.Tensor:
        """Training step with a single batch."""
        if not self.sample_data:
            raise ValueError("sample_data is not initialized.")
        
        sample_data = self.sample(self.sample_data, self.c.n_equil_step)

        self.log("acc", sample_data.get('acc'), on_epoch= True)
        self.log("deltar", sample_data.get('deltar'), on_epoch= True)
        
        data = sample_data.get('data')
        
        out = self(data)
        loss = self.loss_fn(out)

        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer for the model."""
        return optim.Adam(self.parameters(), lr= self.c.lr)
    



# dep 


# class CustomTrainer(pl.LightningModule):
# 	"""Custom model class."""

# 	def __init__(
# 	    self, 
# 	    model: Model,
#         opt: Opt,
#     ):
# 		super().__init__()
# 		self.c = c
# 		self.model = nn.Sequential(
# 			# Add your model layers here
# 		)
# 		self.loss_function = nn.CrossEntropyLoss()
# 		self.train_acc = torchmetrics.Accuracy()
# 		self.val_acc = torchmetrics.Accuracy()

# 	def forward(self, x: torch.Tensor) -> torch.Tensor:
# 		return self.model(x)

# 	def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
# 		x, y = batch
# 		y_hat = self(x)
# 		loss = self.loss_function(y_hat, y)
# 		self.train_acc(y_hat.softmax(dim=-1), y)
# 		self.log("train_loss", loss)
# 		self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
# 		return loss

# 	def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
# 		x, y = batch
# 		y_hat = self(x)
# 		loss = self.loss_function(y_hat, y)
# 		self.val_acc(y_hat.softmax(dim=-1), y)
# 		self.log("val_loss", loss)
# 		self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
# 		return loss

# 	def configure_optimizers(self) -> optim.Optimizer:
# 		return optim.Adam(self.parameters(), lr=self.c.lr)

