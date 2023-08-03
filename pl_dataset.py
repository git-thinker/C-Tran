from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
import dataset
import torchvision
import pytorch_lightning as pl
from typing import *

class COCO(pl.LightningDataModule):
    def __init__(
        self, 
        lmdb_path: str,
        mode_json_paths: Dict[str, str],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        train_transforms=None, 
        val_transforms=None, 
    ):
        self.save_hyperparameters()
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        if self.train_transforms is None:
            self.train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: x if x.size(0) == 3 else x.repeat(3, 1, 1)),
                    torchvision.transforms.Normalize(
                        mean=[0.4711, 0.4475, 0.4080],
                        std=[0.2647, 0.2647, 0.2802],
                    )
                ]
            )
        if self.val_transforms is None:
            self.val_transforms = self.train_transforms
        self.dataset = dataset.COCOlmdb(
            lmdb_path=lmdb_path,
            mode_json_paths=mode_json_paths,
            mode='train'
        )
        for mode in mode_json_paths:
            self.dataset.change_mode(mode)
        self.dataset.change_mode('train')
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def prepare_data_per_node(self) -> None:
        return None
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.dataset.change_mode('train')
        self.dataset.transform = self.train_transforms
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.dataset.change_mode('val')
        self.dataset.transform = self.val_transforms
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
    
class VOC(pl.LightningDataModule):
    def __init__(
        self, 
        root: Optional[str]=None, 
        batch_size: int = 128,
        train_transforms: Optional[torch.nn.Module]=None, 
        val_transforms: Optional[torch.nn.Module]=None, 
        test_transforms: Optional[torch.nn.Module]=None, 
    ):
        self.save_hyperparameters()
        self.batch_size = batch_size
        if root:
            self.train_dataset = dataset.VOCDataset(root=root, transform=train_transforms, partition='train')
            self.val_dataset = dataset.VOCDataset(root=root, transform=val_transforms, partition='val')
            self.test_dataset = dataset.VOCDataset(root=root, transform=test_transforms, partition='val')
        else:
            self.train_dataset = dataset.VOCDataset(transform=train_transforms, partition='train')
            self.val_dataset = dataset.VOCDataset(transform=val_transforms, partition='val')
            self.test_dataset = dataset.VOCDataset(transform=test_transforms, partition='val')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.test_dataloader, batch_size=self.batch_size)
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def prepare_data_per_node(self) -> None:
        return None