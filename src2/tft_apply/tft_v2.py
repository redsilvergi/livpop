#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:46:48 2025

@author: eungi
"""


import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


datelist = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]
yrmthlist = [
    "202404",
    # "202403",
    # "202402",
    # "202401",
    # "202312",
    # "202311",
    # "202310",
    # "202309",
    # "202308",
    # "202307",
    # "202306",
    # "202305",
]
# yrmthlist = ["202404"]


def handle_yrmthd(yrmth):
    if yrmth[4:] == "01":
        return 31
    elif yrmth[4:] == "02":
        return 28
    elif yrmth[4:] == "03":
        return 31
    elif yrmth[4:] == "04":
        return 30
    elif yrmth[4:] == "05":
        return 31
    elif yrmth[4:] == "06":
        return 30
    elif yrmth[4:] == "07":
        return 31
    elif yrmth[4:] == "08":
        return 31
    elif yrmth[4:] == "09":
        return 30
    elif yrmth[4:] == "10":
        return 31
    elif yrmth[4:] == "11":
        return 30
    elif yrmth[4:] == "12":
        return 31


checklist = []
for yrmth in yrmthlist:
    for i in range(handle_yrmthd(yrmth) - 14, handle_yrmthd(yrmth)):
        x = yrmth + datelist[i]
        checklist.append(x)

print(checklist)
len(checklist)

ymd = "20230501"
df = pd.read_parquet(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-{ymd[:6]}/processed/LOCAL_PEOPLE_{ymd}.parquet")

unique_ids = df["census_id"].unique()

for ymd in checklist:
    print(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-202404/processed/LOCAL_PEOPLE_{ymd}.parquet")

# data fetch and process ----------------------------------------------------------------------
missing_hours_list = []
for ymd in checklist:
    df = pd.read_parquet(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-{ymd[:6]}/processed/LOCAL_PEOPLE_{ymd}.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    hour_counts = df.groupby("census_id")["hour"].nunique().reset_index(name="hour_count")
    missing_hours = hour_counts[hour_counts["hour_count"] != 24]
    missing_hours_list.append(missing_hours)


missing_hours_df = pd.concat(missing_hours_list).reset_index(drop=True)
unique_ids_mss = missing_hours_df["census_id"].unique()

target_ids = np.setdiff1d(unique_ids, unique_ids_mss)
desired_count = round(len(target_ids) * 0.05)
np.random.seed(42)
selected_ids = np.random.choice(list(target_ids), size=desired_count, replace=False)

ids = ["census_id"]  # ["adm_id", "census_id"]
labels = ["xpop_total"]
earliest_time_str = "2023-04-17 00:00:00"
earliest_time = pd.to_datetime(earliest_time_str)
df_list = []
for ymd in checklist:
    df = pd.read_parquet(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-{ymd[:6]}/processed/LOCAL_PEOPLE_{ymd}.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df.index = df["time"]
    # df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    for label in labels:
        df2 = df[df[ids[0]].isin(selected_ids)]
        tmp = pd.DataFrame({"xpop": df2[label]})
        tmp = pd.concat([tmp, df2[ids]], axis=1)
        date = tmp.index
        tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
        tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["date"] = date
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month
        df_list.append(tmp)


print(df_list)

time_df = pd.concat(df_list).reset_index(drop=True)

meancheck = time_df[["census_id", "xpop"]].groupby(["census_id"]).mean()
meancheck


# ----------------------------------------------------------------------

max_prediction_length = 24
max_encoder_length = 7 * 24

training_cutoff = time_df["hours_from_start"].max() - max_prediction_length


training = TimeSeriesDataSet(
    time_df[lambda x: x.hours_from_start <= training_cutoff],
    time_idx="hours_from_start",  # The time column is hours_from_start, which acts as the timeline.
    target="xpop",  # The model is predicting xpop, the main target variable.
    group_ids=["census_id"],  # The dataset contains multiple consumers, and the model will learn patterns separately for each census_id.
    min_encoder_length=max_encoder_length // 2,  # The model will use at least half of max_encoder_length (168/2 = 84 hours) of past data.
    max_encoder_length=max_encoder_length,  # The maximum past data used is 168 hours (7 days).
    min_prediction_length=1,  # The model must predict at least 1 future time step.
    max_prediction_length=max_prediction_length,  # The model can predict up to 24 time steps (hours) ahead.
    static_categoricals=["census_id"],  # census_id is a categorical variable that remains constant for each time series.
    time_varying_known_reals=[
        "hours_from_start",
        "day",
        "day_of_week",
        "month",
        "hour",
    ],  # These are known in advance and do not depend on predictions
    time_varying_unknown_reals=["xpop"],
    target_normalizer=GroupNormalizer(
        groups=["census_id"], transformation="softplus"
    ),  # Normalizes xpop values separately for each census_id group. #Uses SoftPlus transformation, which ensures positive values and smooths out variations.
    add_relative_time_idx=True,  # Adds a relative time index (helps the model generalize across different sequences).
    add_target_scales=True,  # Adds scaling parameters for xpop, allowing the model to denormalize outputs.
    add_encoder_length=True,  # Adds the encoder sequence length as a feature, so the model knows how much past data is available.
    # allow_missing_timesteps=True,  # check later
)

######################### WHY MISSING TIMESTEPS? PROCESSED

# This function clones an existing TimeSeriesDataSet (in this case, training) but applies it to a new dataframe (time_df).
# It ensures that the validation dataset has the same transformations, normalizations, and settings as the training dataset.
# This is the original full dataset (which includes both training and validation data).
# from_dataset will filter this dataframe appropriately based on the predict=True argument.
# predict(True) Purpose: It tells the function that this dataset will be used for prediction (not training).
# - The model will not use actual xpop values as targets.
# - Instead, it will treat the xpop column as unknown future values to be predicted.
validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

# check for num_workers----------------------------------------------------------------------
import os
import multiprocessing
import torch

print("CPU cores:", os.cpu_count())
# 11
print("CPU cores:", multiprocessing.cpu_count())
# 11
print(torch.backends.mps.is_available())
# True

# create dataloaders for  our model ----------------------------------------------------------------------
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
# DataLoaders handle batching, shuffling, and loading data into memory during training.
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=11, persistent_workers=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=11, persistent_workers=True)


# Baseline
from pytorch_forecasting.models import Baseline


# Ensure both tensors are on the same device ----------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available, else CPU
device
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)]).to(device)


class MyBaseline(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the hyperparameter saving to ignore these attributes.
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])


baseline_model = MyBaseline()
baseline_predictions = baseline_model.predict(val_dataloader).to(device)
mae = (actuals - baseline_predictions).abs().mean().item()
print("Baseline MAE:", mae)
# GPU available: True (mps), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# Baseline MAE: 241.35018920898438

# v1_result ----------------------------------------------------------------------
# /Users/eungi/Desktop/eungi_dt/mim/livpop/env2/lib/python3.12/site-packages/lightning/pytorch/utilities/parsing.py:209: Attribute 'loss' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['loss'])`.
# /Users/eungi/Desktop/eungi_dt/mim/livpop/env2/lib/python3.12/site-packages/lightning/pytorch/utilities/parsing.py:209: Attribute 'logging_metrics' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['logging_metrics'])`.
# GPU available: True (mps), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# Baseline MAE: 162.53802490234375


# trainer ----------------------------------------------------------------------
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# import lightning.pytorch as pl
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs_v2")

trainer = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    limit_train_batches=0.5,  # Use 10% of training batches
    limit_val_batches=0.5,  # Use 10% of validation batches
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.1,
    hidden_size=6,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=6,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
    optimizer="adam",
)

# train ----------------------------------------------------------------------
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
# best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# print(best_tft)

# load saved best model
best_model_path = (
    "/Users/eungi/Desktop/eungi_dt/mim/livpop/src2/tft_apply/res/lightning_logs/lightning_logs/version_0/checkpoints/epoch=8-step=43227.ckpt"
)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Start tensorboard
# tensorboard --logdir=lightning_logs2/lightning_logs

# predict ----------------------------------------------------------------------
device
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to(device)
actuals

predictions = best_tft.predict(val_dataloader)
# predictions22 = best_tft.predict(val_dataloader, return_y=True)
# predictions22[4]
predictions.shape

# average p50(quantile median) loss overall
print((actuals - predictions).abs().mean().item())
# 18.50079917907715

# average p50 loss per time series
print((actuals - predictions).abs().mean(axis=1))
# tensor([23.3396, 40.0890,  3.1956, 28.1838, 16.5959, 44.8896, 15.1955, 29.0657,
#         28.8332,  7.9187, 39.5820, 35.6537, 20.6536, 23.0582,  9.1013,  4.3239,
#         22.3858, 11.4168, 26.7229,  1.4126,  2.8211,  1.4372, 14.2154, 29.2589,
#          5.5356, 15.8859,  3.4669, 26.9642,  8.9944, 30.5899, 10.9652, 16.4821,
#          8.0071, 34.7888,  6.4980], device='mps:0')

# ----------------------------------------------------------------------
# (1.0765 +  7.6859 +  2.4955 + 6.9401 + 14.6207)/5
# tensor([ 1.0765,  7.6859,  2.4955,  6.9401, 14.6207], device='mps:0')
# The last 2 time-series have a bit higher loss because their relative magnitude is also high.
# ----------------------------------------------------------------------


# raw predict ----------------------------------------------------------------------
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
print(raw_predictions._fields)
print(raw_predictions.output._fields)
print(raw_predictions.output.prediction.shape)
# print(raw_predictions.output.prediction)

# We get predictions of [desired_count] time-series for 24 hours.
# For each hour we get 7 predictions - these are the 7 quantiles:
# [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
# We are mostly interested in the 4th quantile which represents, let's say, the 'median loss'
# fyi, although docs use the term quantiles, the most accurate term are percentiles

# plot_prediction ----------------------------------------------------------------------
import matplotlib.pyplot as plt

for idx in range(desired_count):
    print(idx)
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(
        raw_predictions.x,
        raw_predictions.output,
        idx=idx,
        add_loss_to_title=QuantileLoss(),
        ax=ax,
    )


# select test_ids ----------------------------------------------------------------------

test_ids = np.setdiff1d(target_ids, selected_ids)
# test_count = 2
test_count = round(len(target_ids) * 0.003)
test_count
np.random.seed(42)
test_ids2 = np.random.choice(list(test_ids), size=test_count, replace=False)
test_ids2

# data fetch and process for test_ids2 ----------------------------------------------------------------------

# ids = ["census_id"]  # ["adm_id", "census_id"]
# labels = ["xpop_total"]
# earliest_time_str = "2023-05-01 00:00:00"
# earliest_time = pd.to_datetime(earliest_time_str)
df_list2 = []
for ymd in checklist:
    df = pd.read_parquet(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-{ymd[:6]}/processed/LOCAL_PEOPLE_{ymd}.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df.index = df["time"]
    # df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    for label in labels:
        df2 = df[df[ids[0]].isin(test_ids2)]
        tmp = pd.DataFrame({"xpop": df2[label]})
        tmp = pd.concat([tmp, df2[ids]], axis=1)
        date = tmp.index
        tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
        tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["date"] = date
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month
        df_list2.append(tmp)


print(df_list2)

time_df2 = pd.concat(df_list2).reset_index(drop=True)
time_df2.info()

meancheck = time_df2[["census_id", "xpop"]].groupby(["census_id"]).mean()
meancheck

from pytorch_forecasting.data.encoders import NaNLabelEncoder

test_ts_set = TimeSeriesDataSet.from_dataset(
    training,
    time_df2,
    predict=True,
    stop_randomization=True,
    categorical_encoders={"census_id": NaNLabelEncoder(add_nan=True)},
)
# test_ts_set = TimeSeriesDataSet.from_dataset(training, time_df2, predict=True, stop_randomization=True)

test_dataloader = test_ts_set.to_dataloader(
    train=False,
    batch_size=batch_size * 10,
    num_workers=10,
    persistent_workers=True,  # Adjust batch size as needed
)

new_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)
print(new_predictions._fields)
print(new_predictions.output._fields)
print(new_predictions.output.prediction.shape)


import matplotlib.pyplot as plt

for idx in range(53):
    print(idx)
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(
        new_predictions.x,
        new_predictions.output,
        idx=idx,
        add_loss_to_title=QuantileLoss(),
        ax=ax,
    )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Assume df is your DataFrame and 'time' is a column with datetime info.
# Convert the time column to datetime if not already


# df_list2 = []
# for df in df_list:
#     df["time"] = pd.to_datetime(df["time"])
#     df.index = df["time"]
#     df.sort_index(inplace=True)
#     earliest_time_list.append(df.index.min())
#     df_list2.append(df)


# def lbltrans(label):
#     if label == "xpop_total":
#         return 0
#     elif label == "xpop_m_0to9":
#         return 1
#     elif label == "xpop_m_10to19":
#         return 2
#     else:
#         return None


# ids = ["census_id"]  # ["adm_id", "census_id"]
# labels = [
#     "xpop_total",
#     "xpop_m_0to9",
#     "xpop_m_10to19",
# ]  # xpop_id = ["xpop_total", "xpop_m_0to9", "xpop_m_10to19"] [0,1,2]

# df_list3 = []
# for df in df_list2:
#     tmplist = []
#     for label in labels:
#         tmp = pd.DataFrame({"xpop": df[label]})
#         tmp = pd.concat([tmp, df[ids]], axis=1)
#         date = tmp.index
#         tmp["xpop_id"] = lbltrans(label)
#         tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (
#             date - earliest_time
#         ).days * 24
#         tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")
#         tmp["days_from_start"] = (date - earliest_time).days
#         tmp["date"] = date
#         tmp["hour"] = date.hour
#         tmp["day"] = date.day
#         tmp["day_of_week"] = date.dayofweek
#         tmp["month"] = date.month

#         tmplist.append(tmp)
#     # tmplist
#     time_df = pd.concat(tmplist).reset_index(drop=True)
#     # time_df.info()
#     df_list3.append(time_df)

# meancheck = time_df[["census_id", "xpop_id", "xpop"]].groupby(["census_id", "xpop_id"]).mean()


# ----------------------------------------------------------------------
import pandas as pd


df_list = []
missing_hours_list = []
df = pd.read_parquet("/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-202404/processed/LOCAL_PEOPLE_20240403.parquet")


# Assume df is your DataFrame and 'time' is a column with datetime info.
# Convert the time column to datetime if not already
df["time"] = pd.to_datetime(df["time"])

# Extract hour from the datetime column
df["hour"] = df["time"].dt.hour

# Group by census_id and count unique hours
hour_counts = df.groupby("census_id")["hour"].nunique().reset_index(name="hour_count")

# Find census_ids that do not have all 24 hours
missing_hours = hour_counts[hour_counts["hour_count"] != 24]

df_list.append(df)
missing_hours_list.append(missing_hours)

print("Census IDs missing some hours:")
print(missing_hours)

df.isna().sum()
df.info()

df[df[""]]

# ----------------------------------------------------------------------
tmp = df[df["census_id"] == 1103063010109]
tmp = df[df["census_id"].astype(str) == "1103063010109"]
tmp

ymd = "20230501"
df = pd.read_parquet(f"/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-{ymd[:6]}/processed/LOCAL_PEOPLE_{ymd}.parquet")
ids = ["census_id"]
unique_ids = df["census_id"].unique()
desired_count = round(len(unique_ids) * 0.002)  # for example, select 100 unique IDs
selected_ids = np.random.choice(unique_ids, size=desired_count, replace=False)
