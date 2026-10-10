# Written by: Inal A. Mashukov
# MIT License.

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import argparse
import json
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.transformer import ORACLE
from src.utils import setup_logging, log_model_stats, get_benchmark_indices

class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0, path='checkpoint.pth', trace_func=logging.info):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, revin_layer=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, revin_layer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, revin_layer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, revin_layer):
        checkpoint = {'model_state_dict': model.state_dict()}
        if revin_layer is not None:
            checkpoint['revin_state_dict'] = revin_layer.state_dict()
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


class ForecastDataset(Dataset):
    def __init__(self, path: str, lookback: int = 128, horizon: int = 32) -> None:
        df = pd.read_csv(path)
        df = df.drop(columns=['date']) if 'date' in df.columns else df.iloc[:, 1:]
        self.raw_data = torch.from_numpy(df.values.astype(np.float32))
        
        self.lookback = lookback
        self.horizon = horizon
        
        dataset_name = path.lower()
        raw_len = len(self.raw_data)
        self.train_raw_end = 8640 if "etth" in dataset_name else (34560 if "ettm" in dataset_name else int(raw_len * 0.7))

        # Statically compute normalization statistics based exclusively on the training split
        train_data = self.raw_data[:self.train_raw_end]
        self.mean = train_data.mean(dim=0, keepdim=True)
        self.std = train_data.std(dim=0, keepdim=True) + 1e-6
        
        # Normalize the entire dataset once
        normalized_data = (self.raw_data - self.mean) / self.std
        self.x, self.y = [], []
        total_len = self.lookback + self.horizon 
        for i in range(len(normalized_data) - total_len + 1):
            self.x.append(normalized_data[i : i + self.lookback])
            self.y.append(normalized_data[i + self.lookback : i + total_len])

    def get_raw_lookback(self, idx: int) -> torch.Tensor:
        return self.raw_data[idx : idx + self.lookback]

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm': return self._denormalize(x)
        else:
            raise ValueError(f"Must use 'norm' or 'denorm', not {mode}. \n")
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        return x * self.affine_weight + self.affine_bias if self.affine else x
    def _denormalize(self, x):
        if self.affine: x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        return (x * self.stdev) + self.mean
    

def build_model_and_optimizer(args, model_config, train_config, inferred_num_features, num_classes, forecast_len, is_pretrain_task, device):
    model = ORACLE(
        lookback=model_config["lookback"],
        patch_size=model_config["patch_size"],
        stride=model_config["stride"],
        d_model=model_config["d_model"],
        dim_feedforward=model_config["dim_feedforward"],
        nheads=model_config["nheads"],
        num_layers=model_config["num_layers"],
        num_classes=num_classes,
        forecast_len=forecast_len,
        pretrain=is_pretrain_task
    ).to(device)
    
    if args.pretrained_ckpt is not None:
        logging.info(f"Loading pretrained weights from {args.pretrained_ckpt}")
        checkpoint = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        model.load_state_dict(state_dict, strict=False)

    revin_layer = RevIN(num_features=inferred_num_features).to(device) if args.task in ["forecasting", "pretraining"] else None
    optim_params = list(model.parameters())
    if revin_layer is not None and revin_layer.affine: optim_params += list(revin_layer.parameters())

    optimizer = optim.AdamW(optim_params, lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
    
    return model, revin_layer, optimizer


def run_training_loop(model, revin_layer, optimizer, train_loader, val_loader, args, train_config, device, is_pretrain_task, use_time, use_feature, criterion, results_dir, desc_prefix="Base"):
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config["learning_rate"], epochs=train_config["epochs"], steps_per_epoch=len(train_loader), pct_start=0.3)
    early_stopping = EarlyStopping(patience=train_config.get("early_stopping_patience", 5), verbose=True, path=os.path.join(results_dir, f"{desc_prefix}_checkpoint.pth"))

    for epoch in range(train_config["epochs"]):
        model.train()
        if revin_layer is not None: revin_layer.train()
        
        loader = tqdm(train_loader, desc=f"{desc_prefix} Ep {epoch+1}/{train_config['epochs']}", leave=False)
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'): 
                if is_pretrain_task:
                    bx_norm = revin_layer(bx, mode='norm') if revin_layer is not None else bx
                    pred, target, mask = model(bx_norm, use_time=use_time, use_feature=use_feature, mask_ratio=args.mask_ratio)
                    loss = (pred - target) ** 2
                    loss = loss.mean(dim=-1)
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                else:
                    by = by.to(device, non_blocking=True)
                    if args.task == "forecasting":
                        bx_norm = revin_layer(bx, mode='norm')
                        outputs = model(bx_norm, use_time=use_time, use_feature=use_feature).permute(0, 2, 1)
                        outputs = revin_layer(outputs, mode='denorm')
                        loss = criterion(outputs, by)
                    else:
                        outputs = model(bx, use_time=use_time, use_feature=use_feature)
                        loss = criterion(outputs, by if args.task != "regression" else by.float().view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loader.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Block
        model.eval()
        if revin_layer is not None: revin_layer.eval()
        val_loss, val_samples = 0.0, 0
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                if is_pretrain_task:
                    bx_norm = revin_layer(bx, mode='norm') if revin_layer is not None else bx
                    pred, target, mask = model(bx_norm, use_time=use_time, use_feature=use_feature, mask_ratio=args.mask_ratio)
                    loss = (pred - target) ** 2
                    loss = loss.mean(dim=-1)
                    batch_loss = ((loss * mask).sum() / mask.sum().clamp(min=1)).item()
                else:
                    by = by.to(device, non_blocking=True)
                    if args.task == "forecasting":
                        bx_norm = revin_layer(bx, mode='norm')
                        outputs = model(bx_norm, use_time=use_time, use_feature=use_feature).permute(0, 2, 1)
                        outputs = revin_layer(outputs, mode='denorm')
                        batch_loss = criterion(outputs, by).item()
                    else:
                        outputs = model(bx, use_time=use_time, use_feature=use_feature)
                        batch_loss = criterion(outputs, by if args.task != "regression" else by.float().view(-1)).item()
                
                val_loss += batch_loss * bx.size(0); val_samples += bx.size(0)

        avg_val_loss = val_loss / val_samples
        logging.info(f"{desc_prefix} Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        early_stopping(avg_val_loss, model, revin_layer)
        if early_stopping.early_stop: break

    checkpoint = torch.load(os.path.join(results_dir, f"{desc_prefix}_checkpoint.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    if revin_layer is not None and 'revin_state_dict' in checkpoint: 
        revin_layer.load_state_dict(checkpoint['revin_state_dict'])

    return model, revin_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, required=True, choices=["forecasting", "pretraining"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="ORACLE_Run")
    parser.add_argument("--disable_time", action="store_true")
    parser.add_argument("--disable_feature", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.40)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    args = parser.parse_args()

    use_time = not args.disable_time
    use_feature = not args.disable_feature

    if args.task in ["forecasting", "pretraining"]:
        random.seed(2021); np.random.seed(2021); torch.manual_seed(2021)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(2021)

    setup_logging(args.run_name)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    results_dir = os.path.join("results", args.run_name); os.makedirs(results_dir, exist_ok=True)

    with open("./config.json", "r") as f: config = json.load(f)
    train_config, model_config = config["train"], config["model"]
    
    with open(os.path.join(results_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=4)

    logging.info(f"Pipeline initialization for: {args.task.upper()} | Target Config Node: {args.data}")
    
    if args.task in ["forecasting"]:
        full_dataset = ForecastDataset(path=args.data, lookback=model_config["lookback"], horizon=model_config["horizon"])
        inferred_num_features = full_dataset.raw_data.shape[-1]
        train_indices, val_indices, test_indices = get_benchmark_indices(len(full_dataset.raw_data), args.data, model_config["lookback"], model_config["horizon"])
        
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
        
        train_loader = DataLoader(train_set, batch_size=train_config["batch_size"], shuffle=True, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=train_config["batch_size"], shuffle=False, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=train_config["batch_size"], shuffle=False, pin_memory=True, drop_last=False)
        
        num_classes, forecast_len = None, model_config["horizon"]
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Task must be 'forecasting', not {args.task} \n")

    is_pretrain_task = (args.task == "pretraining")

    model, revin_layer, optimizer = build_model_and_optimizer(
        args, model_config, train_config, inferred_num_features, num_classes, forecast_len, is_pretrain_task, device
    )

    log_model_stats(model, device)

    # ---------------------------------------------------------
    # Training Stage
    # ---------------------------------------------------------
    logging.info("=" * 80)
    logging.info("Commencing Training Phase...")
    logging.info("=" * 80)
    model, revin_layer = run_training_loop(
        model=model,
        revin_layer=revin_layer,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        train_config=train_config,
        device=device,
        is_pretrain_task=is_pretrain_task,
        use_time=use_time,
        use_feature=use_feature,
        criterion=criterion,
        results_dir=results_dir,
        desc_prefix="Train"
    )

    # ---------------------------------------------------------
    # Testing Phase
    # ---------------------------------------------------------
    logging.info("=" * 80)
    logging.info("Initiating Standard Test Evaluation Phase...")
    logging.info("=" * 80)

    model.eval()
    if revin_layer is not None: revin_layer.eval()

    mae_loss = nn.L1Loss()
    total_mse, total_mae, sample_count = 0.0, 0.0, 0

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                if is_pretrain_task:
                    bx_norm = revin_layer(bx, mode='norm') if revin_layer is not None else bx
                    pred, target, mask = model(bx_norm, use_time=use_time, use_feature=use_feature, mask_ratio=args.mask_ratio)
                    loss_matrix = (pred - target) ** 2
                    loss_matrix = loss_matrix.mean(dim=-1)
                    current_loss = ((loss_matrix * mask).sum() / mask.sum().clamp(min=1)).item()
                    total_mse += current_loss * bx.size(0)
                else:
                    bx_norm = revin_layer(bx, mode='norm')
                    outputs = model(bx_norm, use_time=use_time, use_feature=use_feature).permute(0, 2, 1)
                    outputs = revin_layer(outputs, mode='denorm')
                    
                    current_loss = criterion(outputs, by).item()
                    current_mae = mae_loss(outputs, by).item()

                    total_mse += current_loss * bx.size(0)
                    total_mae += current_mae * bx.size(0)

            sample_count += bx.size(0)

    if is_pretrain_task:
        logging.info(f"FINAL TEST PERFORMANCE | Evaluation Reconstruction Loss: {total_mse/sample_count:.4f}")
    else:
        logging.info(f"FINAL TEST PERFORMANCE | MSE: {total_mse/sample_count:.4f} | MAE: {total_mae/sample_count:.4f}")

if __name__ == "__main__":
    main()
