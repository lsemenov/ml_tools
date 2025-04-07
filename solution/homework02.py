#!/usr/bin/env python
import argparse
from io import BytesIO
from zipfile import ZipFile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import torch.utils.data as data
import os

import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as pl
from lightning import Trainer, LightningModule, LightningDataModule# Джентельменский наборчик
from lightning.pytorch.callbacks import  ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# your code here
@dataclass
class CFG:
    seed: int = 2025
    accelerator: str = "auto"  # Автоматическое определение
    devices: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 200
    test_size: float = 0.2
    num_workers: int = min(2, os.cpu_count())
    epochs: int = 10
    gpus: int = 2
    stride: int = 1
    dilation: int =1
    n_classes: int =25
        
    def __post_init__(self):
        pl.seed_everything(self.seed)
        # Автоматическое определение доступных устройств
        if torch.cuda.is_available():
            self.accelerator = "gpu"
            self.devices = torch.cuda.device_count()
        else:
            self.accelerator = "cpu"
            self.devices = 1

# Dataset
class SignLanguageDataset(data.Dataset):

    
    def __init__(self, df, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        
        self.df = df
        self.transform = transform
        self.is_train = is_train

        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        label = self.df.iloc[index, 0]
        
        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        if self.transform is not None:
            img =self.transform(img)
        
        return img, label

# Data Module
class SignLanguageDatasetLightning(LightningDataModule):
    
    def __init__(self, cfg: CFG):
        super().__init__()
        
        self.cfg = cfg
        self.train_transform = transforms.Compose([
        #transforms.Normalize(159, 40),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
    ])
            
    def prepare_data(self):
        """Download and prepare data (called once)"""
        self.train_df = self._read_zip_csv(
            "https://github.com/a-milenkin/ml_instruments/raw/main/data/sign_mnist_train.csv.zip"
        )
        self.test_df = self._read_zip_csv(
            "https://github.com/a-milenkin/ml_instruments/raw/main/data/sign_mnist_test.csv.zip"
        )



    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df, val_df = train_test_split(
                self.train_df, 
                test_size=self.cfg.test_size,
                stratify=self.train_df.iloc[:, 0]
            )
            
            self.train = SignLanguageDataset(train_df, self.train_transform)
            self.val = SignLanguageDataset(val_df, is_train=False)
            
        if stage == "test" or stage is None:
            self.test = SignLanguageDataset(self.test_df, is_train=False)
                     
    def _make_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
    
    def _read_zip_csv(self, url: str) -> pd.DataFrame:
        """Helper to read CSV from ZIP"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise

        with ZipFile(BytesIO(response.content)) as zip_file:
            csv_files = [f for f in zip_file.namelist() 
                        if f.endswith('.csv') and '__MACOSX' not in f]
            if len(csv_files) != 1:
                raise ValueError(f"Expected 1 CSV, found {len(csv_files)}")
                
            with zip_file.open(csv_files[0]) as f:
                return pd.read_csv(f)
            
    def train_dataloader(self)-> DataLoader:
        # Возвращаем Train Dataloader
        return self._make_dataloader(self.train)
    
    def val_dataloader(self)-> DataLoader:
        # Возвращаем Valid Dataset
        return self._make_dataloader(self.val)
        
    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test)
    
# Model
class MyConvNetLightning(LightningModule):
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.lr

        self.stride = cfg.stride
        self.dilation = cfg.dilation
        self.n_classes = cfg.n_classes

        
        self.model = nn.Sequential(
            #input=(batch, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(8),
            # (batch, 8, 28, 28)
            nn.AvgPool2d(2),
            # (batch, 8, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(16),
            # (batch, 16, 14, 14)
            nn.AvgPool2d(2),
            # (batch, 16, 7, 7)
            nn.ReLU()
            )
        
        self.lin1 = nn.Linear(in_features=16*7*7, out_features=100)
        # (batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=cfg.dropout)
        self.lin2 = nn.Linear(100, self.n_classes)
        # (batch, 25)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        
        x = self.model(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        
        return x
    
    def _shared_step(self, batch: tuple, step: str):
        # получаем данные
        data, target = batch
        
        pred = self(data)
        
        loss= self.criterion(pred, target)
        acc = (pred.argmax(dim=1) == target).float().mean()
        
        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/acc": acc
        }
        
        self.log_dict(loss_dict, prog_bar=True)
        
        if step == "val":
            self.log("val/acc", acc, prog_bar=True)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch,  "train")
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch,"val")
        return loss
    
    def test_step(self, batch, batch_idx ):
        loss = self._shared_step(batch, "test")
        return loss
     

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2, # Сколько эпох ждать без улучшения
            verbose=True, # Печатать, когда LR меняется
            factor=0.5, # Насколько уменьшить LR (в 2 раза, 0.1 = в 10 раз)
            mode='min', # Что хотим минимизировать (например, val_loss)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss" # Должно совпадать с логгируемым именем
            }
        }
    
def main(args, cfg):

    dataset = SignLanguageDatasetLightning(cfg)
    dataset.prepare_data()
    dataset.setup("fit")
    
    model = MyConvNetLightning(cfg)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best-{epoch}-{val_acc:.2f}",
        save_top_k=1
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.epochs, 
        log_every_n_steps=1,
        devices=cfg.devices if cfg.accelerator == "gpu" else "auto",
        # отключение чекпоинтов в режиме разработки
        enable_checkpointing=not args.fast_dev_run,
        fast_dev_run=args.fast_dev_run
        
    )
    
    if args.fast_dev_run:
        try:
#             trainer.test(datamodule=dataset, ckpt_path="best")
            trainer.validate(model, datamodule=dataset)
            print("Тестовый прогон успешно пройден")
        except Exception as err:
            print(f"Тестовый прогон завершился с ошибкой {err}")
            return
    else:
        # Полный цикл обучения + тестирование
        trainer.fit(model, datamodule=dataset)
        trainer.test(model, datamodule=dataset) 
        
    
    # Инференс на одном образце
    dataset.setup("test")
    model.eval()
    sample_img, true_label = dataset.test[0]
    sample = sample_img.unsqueeze(0)  # берём первый образец из теста
    with torch.no_grad():
        pred = model(sample)
        predicted_class = pred.argmax(dim=1).item()
        print(f"Инференс на одном примере: Fact: {true_label}, Prediction:{predicted_class}")

        # Сохраним модель
    state = {'model': model.state_dict(),
            'epoch': 20}

    model_path = Path('../models')
    model_name = 'myconvnet_sign_lang.pth'
    model_path.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее не существует

    torch.save(state, model_path / model_name)

if __name__ == "__main__":
    # your code here
    parser = argparse.ArgumentParser()
    
    # определён через action="store_true". При таком способе использования флаг устанавливается в True
    # , если он присутствует, и не требует передачи дополнительного значения. 
    # То есть, для включения режима fast_dev_run достаточно указать просто флаг без параметров.
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
                                                                                       
    cfg = CFG()
  
    main(args, cfg)
