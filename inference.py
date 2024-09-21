import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import xarray as xr

from datamodule import ERA5DataModule
from gravity_wave_model import UNetWithTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
device = f"cuda:{local_rank}"
dtype = torch.float32

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(model,ckpt_singular):

    print('Loading weights from', ckpt_singular)
    state_dict = torch.load(f=ckpt_singular, map_location=device, weights_only=True)

    ignore_layers = [
        "input_scalers_mu",
        "input_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
        "patch_embedding.proj.weight",
        "patch_embedding_static.proj.weight",
        "unembed.weight",
        "unembed.bias",
        "output_scalers",
    ]

    for layer in ignore_layers:
        state_dict.pop(layer, None)
    model.load_state_dict(state_dict)
    print('Loaded weights')
    return model

def get_model(cfg, vartype,ckpt_singular: str) -> torch.nn.Module:
    model: torch.nn.Module = UNetWithTransformer(
        lr=cfg.lr,
        hidden_channels=cfg.hidden_channels,
        in_channels={"uvtheta122": 366, "uvtp122": 488, "uvtp14": 56}[vartype],
        out_channels={"uvtheta122": 244, "uvtp122": 366, "uvtp14": 42}[vartype],
        n_lats_px=cfg.n_lats_px,
        n_lons_px=cfg.n_lons_px,
        in_channels_static=cfg.in_channels_static,
        mask_unit_size_px=cfg.mask_unit_size_px,
        patch_size_px=cfg.patch_size_px,
        device=device,
    )
    model = DDP(model.to(local_rank, dtype=dtype), device_ids=[local_rank])
    model = load_checkpoint(model,ckpt_singular)

    return model

def get_data(data_path: str, file_glob_pattern: str) -> torch.utils.data.DataLoader:
    datamodule = ERA5DataModule(
        batch_size=8,
        num_data_workers=8,
        train_data_path=None,
        valid_data_path=data_path,
        file_glob_pattern=file_glob_pattern,
    )
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()
    return dataloader

def main(cfg, vartype, ckpt_path: str, data_path: str, results_dir: str, file_glob_pattern: str='*.nc'):
    setup()

    model: torch.nn.Module = get_model(cfg, vartype, ckpt_singular=ckpt_path)
    dataloader: torch.utils.data.DataLoader = get_data(
        data_path=data_path, file_glob_pattern=file_glob_pattern
    )

    # Pre-allocate an xarray.DataArray to store results
    da_results: xr.DataArray = xr.full_like(other=dataloader.dataset.ds.isel(odim=slice(0, model.module.decoder.final_conv.out_channels)).output,fill_value=np.NaN,)
    # assert da_results.sizes == {"time": 744, "odim": 42, "lat": 64, "lon": 128}

    # Main prediction loop
    total: int = len(dataloader)
    pbar = tqdm.tqdm(iterable=enumerate(dataloader), total=total)
    for i, batch in pbar:
        batch = {
            k: v.to(device="cuda") for k, v in batch.items()
        }  # move data to the same device as the model

        with torch.no_grad():
            output: torch.Tensor = model(batch)  # run inference
            # assert output.shape == torch.Size([8, 366, 64, 128])

            # Save input and output tensors to Pytorch format
            # torch.save(output, os.path.join(results_dir, f"output_{i}.pt"))
            # torch.save(batch, os.path.join(results_dir, f"input_batch_{i}.pt"))

            # Save output to NetCDF
            t_start: int = i * dataloader.batch_size
            t_stop: int = t_start + dataloader.batch_size
            t_slice = slice(t_start, t_stop)
            da_results[dict(time=t_slice)] = xr.DataArray(
                data=output.cpu(),
                dims=da_results.dims,
                coords=da_results.isel(time=t_slice).coords,
            )
            if i % 20 == 0 or i == total - 1:  # Output to NetCDF every 20 steps
                da_results.to_netcdf(
                    path=os.path.join(results_dir, "output.nc"),
                    mode="w",  # always rewrite file
                )

            # Report loss
            loss: torch.Tensor = F.mse_loss(input=output, target=batch["target"])
            pbar.set_postfix(
                ordered_dict={
                    "t_slice": f"{t_slice.start}-{t_slice.stop}",  # time slice
                    "loss": loss.item(),
                }
            )


# %%
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="uvtp122",
    )
    parser.add_argument(
        "--ckpt_path",
        default="checkpoints/uvtp122/magnet-flux-uvtp122-epoch-06-loss-0.2274.pt",
    )    
    parser.add_argument(
        "--data_path",
        default="gravity_wave_flux/uvtp122",
    )
    parser.add_argument(
        "--results_dir",
        default="results/uvtp122",
    )
    args = parser.parse_args()


    from config import get_cfg

    cfg = get_cfg()
    os.makedirs(name=args.results_dir, exist_ok=True)
    main(cfg,args.split, args.ckpt_path, args.data_path, args.results_dir,)
    cleanup()
