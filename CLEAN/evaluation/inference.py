# check successful setup
import torch
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.cuda.get_device_name(device))

import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

from torch import from_numpy as np2TT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pathlib import Path
import argparse
import pandas as pd


from CLEAN.utils.build import create_dataset
from CLEAN.evaluation.eval_metrics import calc_eval_metrics
from CLEAN.dataset.sparse_eeg_dataset_val import SparseEEGDataset_val
from CLEAN.dataset.collator_val import SparseEEG_Collator
from CLEAN.model.encoder import Encoder
from CLEAN.model.decoder import DecoderPerceiver
from CLEAN.model.CLEAN import CLEAN

# Create the parser
parser = argparse.ArgumentParser(description="Inference script to generate eval metrics for UPT4EEG.")

# Define the argument
parser.add_argument('--model', type=str, help="Path to .pth model")
parser.add_argument('--config', type=str, help="Config path")
parser.add_argument('--use_montage', type=str, help="Defines which montage to use in inference ('tuh', 'tuh_rand' or 'random').")
parser.add_argument('--save_path', type=str, default='./logs/TUH/CLEAN', help="Save path")
parser.add_argument('--save_model_name', type=str, help="The model name used for saving.", default='CLEAN')

# Parse the arguments
args = parser.parse_args()

model_path = args.model
config_path = args.config
use_montage = args.use_montage
saved_model_type = args.save_model_name
SAVE_PATH = args.save_path

DATASET = 'TUH'    # either 'TUH' or 'BCI' or 'DenoiseNet'
MODEL_CLASS = 'UPT4EEG'
plot_sample = False

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())

d_model = 192*2
dim = 192  
num_heads = 4 #3
depth = 3

SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
window_size = cfg_dataset["window_size"]
stride = cfg_dataset["stride"]



if not os.path.exists(SAVE_PATH):
    try:
        os.makedirs(SAVE_PATH)
    except Exception as e:
        print(f"Failed to create directory '{SAVE_PATH}': {e}")


timestamp = datetime.now().strftime("%b%d_%H-%M-%S")



df_met_per_sub = pd.DataFrame()
mse_list_total = []
r2_list_total = []
snr_db_list_total = []
pcc_list_total = []


for subject in cfg_dataset["subjects_test"]:

    x_test, y_test, ch_names = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        [subject], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
    )

    #cfg_dataset["window_size"]
    #x_train.shape: [Nr of segments, channel nr, sequence length]
    x_test = np2TT(x_test)
    y_test = np2TT(y_test)


    # hyperparameters
    num_supernodes = 512
    input_dim = 1
    output_dim = 1
    use_mlp_posEnc = True



    # initialize model
    model = CLEAN(
        encoder = Encoder(
            input_dim=input_dim,
            ndim=1,
            gnn_dim=d_model,
            enc_dim=dim,
            enc_num_heads=num_heads,
            enc_depth=depth,
            mlp_pos_enc=use_mlp_posEnc,
        ),
        decoder=DecoderPerceiver(
            input_dim=dim,
            output_dim=output_dim,
            ndim=1,
            dim=dim,
            num_heads=num_heads,
            depth=depth,
            mlp_pos_enc=use_mlp_posEnc,
        ),
    )

    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


    state_path = os.path.join(model_path)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])

    query_montage_pairs = [
                ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
                ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
                ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
                ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
                ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
                ('C4', 'P4'), ('P4', 'O2')
                ]
    query = {'query_freq': SFREQ, 'query_montage_pairs': query_montage_pairs}


    test_dataset = SparseEEGDataset_val(x_test, y_test, query, ch_names, cfg_dataset, use_montage=use_montage)

    sample = test_dataset[0]
    print(f"Input features shape: {sample['input_feat'].shape}")
    print(f"Input positions shape: {sample['input_pos'].shape}")
    print(f"Target features shape: {sample['target_feat'].shape}")
    print(f"Output positions shape: {sample['target_pos'].shape}")
    print(f"Query positions shape: {sample['query_pos'].shape}")

    # setup dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        collate_fn=SparseEEG_Collator(num_supernodes=256, deterministic=False),
    )


    ################################## EVALUATION METRICS ##########################
    csv_path = os.path.join(SAVE_PATH, 'eval_metrics_upt4eeg_' + saved_model_type + '_montage_' + use_montage + '.csv')
    csv_summary_path = os.path.join(SAVE_PATH, 'subject_metrics_summary_upt4eeg_' + saved_model_type + '_montage_' + use_montage + '.csv')
    final_summary_path = os.path.join(SAVE_PATH, 'metrics_summary_upt4eeg_'+ saved_model_type + '_montage_' + use_montage + '.csv')

    total_updates = len(test_dataloader)

    test_loss_total = 0
    r2_metric_total = 0
    snr_total = 0
    snr_db_total = 0
    invalid_cnt = 0
    pcc_total = 0
    cc_total = 0
    rmse_total = 0
    rrmse_total = 0
    r2_list = []
    mse_list = []
    snr_list = []
    snr_db_list = []
    pcc_list = []
    cc_list = []
    rrmse_list = []
    rmse_list = []

    pbar = tqdm(total=total_updates)
    pbar.update(0)
    #pbar.set_description(f"MSE loss: {test_loss_total:.4f}, SNR in dB: {snr_total:.4f}, R^2 score: {r2_metric_total:.4f}, PCC: {pcc_total:.4f}, RRMSE: {rrmse_total:.4f}")
        

    for batch in test_dataloader:
        with torch.no_grad():
            y_hat = model(
                input_feat=batch["input_feat"].to(device),
                input_pos=batch["input_pos"].to(device),
                batch_idx=batch["batch_idx"].to(device),
                output_pos=batch["target_pos"].to(device),
            )
        y = batch["target_feat"].to(device)
        eval_metrics = calc_eval_metrics(y_hat, y)
        test_loss_total += eval_metrics['MSE']
        r2_metric_total += eval_metrics['R2']
        pcc_total += eval_metrics['PCC']
        cc_total += eval_metrics['CC']
        rrmse_total += eval_metrics['RRMSE']
        rmse_total += eval_metrics['RMSE']
        snr_total += eval_metrics['SNR']
        snr_db_total += eval_metrics['SNR_dB']
        mse_list.append(eval_metrics['MSE'])
        r2_list.append(eval_metrics['R2'])
        snr_list.append(eval_metrics['SNR'])
        snr_db_list.append(eval_metrics['SNR_dB'])
        pcc_list.append(eval_metrics['PCC'])
        cc_list.append(eval_metrics['CC'])
        rrmse_list.append(eval_metrics['RRMSE'])
        rmse_list.append(eval_metrics['RMSE'])

        pbar.update()
        pbar.set_description(
            f"MSE loss: {eval_metrics['MSE']:.4f}, "
            f"SNR: {eval_metrics['SNR']:.4f}, "
            f"SNR in dB: {eval_metrics['SNR_dB']:.4f}, "
            f"R^2 score: {eval_metrics['R2']:.4f}, "
            f"PCC: {eval_metrics['PCC']:.4f}, "
            f"CC: {eval_metrics['CC']:.4f}, "
            f"RRMSE: {eval_metrics['RRMSE']:.4f}, "
            f"RMSE: {eval_metrics['RMSE']:.4f}"
        )
        
    test_loss_total /= len(test_dataloader)
    r2_metric_total /= len(test_dataloader)
    snr_total /= (len(test_dataloader))
    snr_db_total /= (len(test_dataloader))
    rrmse_total /= len(test_dataloader)
    rmse_total /= len(test_dataloader)
    pcc_total /= len(test_dataloader)
    cc_total /= len(test_dataloader)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header (optional)
        writer.writerow(['MSE', 'R2', 'SNR', 'SNR_dB', 'RRMSE', 'RMSE', 'PCC', 'CC'])
        # Write rows
        for row in zip(mse_list, r2_list, snr_list, snr_db_list, rrmse_list, rmse_list, pcc_list, cc_list):
            writer.writerow(row)


    mse_list_total.extend(mse_list)
    r2_list_total.extend(r2_list)
    snr_db_list_total.extend(snr_db_list)
    pcc_list_total.extend(pcc_list)

    print(f'Data written to {csv_path}')

    print(
        "Average metrics: "
        f"MSE: {test_loss_total}, "
        f"R^2: {r2_metric_total}, "
        f"SNR in dB: {snr_db_total:.4f}, "
        f"PCC: {pcc_total:.4f}, "
    )

    metrics = {"MSE": test_loss_total, 
               "R^2": r2_metric_total,
               "SNR": snr_db_total, 
               "PCC": pcc_total,
               }

    df_met_per_sub[subject[0]] = pd.Series(metrics)  # Updates existing columns or adds new ones


    mse_list = [x for x in mse_list if math.isfinite(x)]
    r2_list = [x for x in r2_list if math.isfinite(x)]
    snr_list = [x for x in snr_list if math.isfinite(x)]

    print(f'Min MSE: {min(mse_list)}, Max MSE: {max(mse_list)}, Min R2: {min(r2_list)}, Max R2: {max(r2_list)}, Min SNR: {min(snr_list)}, Max SNR: {max(snr_list)},')


df_met_per_sub.to_csv(csv_summary_path)

metric_lists = {"MSE": mse_list_total, 
                "R^2": r2_list_total,
                "SNR": snr_db_list_total, 
                "PCC": pcc_list_total,
                }

results = []
for metric_name, values in metric_lists.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    results.append([metric_name, mean, std_dev])

# Write results to a CSV file

with open(final_summary_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Metric", "Mean", "Standard Deviation"])
    # Write metric data
    writer.writerows(results)

print(f"Summary saved to {final_summary_path}")