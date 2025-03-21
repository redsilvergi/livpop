#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:29:56 2025

@author: eungi
"""

# import numpy as np
# import pandas as pd
# from pytorch_forecasting import TimeSeriesDataSet


# check TimeSeriesDataSet ----------------------------------------------------------------------
# sample_data = pd.DataFrame(
#     dict(
#         time_idx=np.tile(np.arange(6), 3),
#         target=np.array([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25, 40, 41, 42, 43, 44, 45]),
#         group=np.repeat(np.arange(3), 6),
#         holidays=np.tile(["X", "Black Friday", "X", "Christmas", "X", "X"], 3),
#     )
# )
# sample_data

# # create the time-series dataset from the pandas df
# dataset = TimeSeriesDataSet(
#     data=sample_data,
#     group_ids=["group"],
#     target="target",
#     time_idx="time_idx",
#     max_encoder_length=2,
#     max_prediction_length=3,
#     time_varying_unknown_reals=["target"],
#     static_categoricals=["holidays"],
#     target_normalizer=None,
# )

# # pass the dataset to a dataloader
# dataloader = dataset.to_dataloader(batch_size=1)

# # load the first batch
# # x, y = next(iter(dataloader))
# print(x["encoder_target"])
# print(x["groups"])
# print("")
# print(x["decoder_target"])
# print(sample_data)

# apply ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

data = pd.read_csv(
    "/Users/eungi/Desktop/eungi_dt/mim/livpop/src2/data/LD2011_2014.txt",
    index_col=0,
    sep=";",
    decimal=",",
)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)
data.head(5)
tmp1 = data.head(5)

data = data.resample("1h").mean().replace(0.0, np.nan)
earliest_time = data.index.min()
df = data[["MT_002", "MT_004", "MT_005", "MT_006", "MT_008"]]
df.head(5)

df_list = []

for label in df:
    print(label)

# test codes in loop
# ts = df["MT_002"]
# start_date = min(ts.ffill().dropna().index)
# end_date = max(ts.bfill().dropna().index)
# active_range = (ts.index >= start_date) & (ts.index <= end_date)
# tmp1 = ts[active_range]
# tmp2 = tmp1.fillna(0.0)
# tmp = pd.DataFrame({"power_usage": tmp2})
# tmp3 = (tmp.index - earliest_time).seconds / 60 / 60
# tmp4 = (tmp.index - earliest_time).days * 24
# tmp5 = (tmp3 + tmp4).astype(int)

# run for loop ----------------------------------------------------------------------
for label in df:

    ts = df[label]

    start_date = min(ts.ffill().dropna().index)
    end_date = max(ts.bfill().dropna().index)

    active_range = (ts.index >= start_date) & (ts.index <= end_date)
    ts = ts[active_range].fillna(0.0)

    tmp = pd.DataFrame({"power_usage": ts})
    date = tmp.index

    tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
    tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")

    tmp["days_from_start"] = (date - earliest_time).days
    tmp["date"] = date
    tmp["consumer_id"] = label
    tmp["hour"] = date.hour
    tmp["day"] = date.day
    tmp["day_of_week"] = date.dayofweek
    tmp["month"] = date.month

    # stack all time series vertically
    df_list.append(tmp)
df_list
time_df = pd.concat(df_list).reset_index(drop=True)

# match results in the original paper
time_df2 = time_df[(time_df["days_from_start"] >= 1096) & (time_df["days_from_start"] < 1346)].copy()

time_df.head(10)
time_df2.head(10)

# mean check
time_df2[["consumer_id", "power_usage"]].groupby("consumer_id").mean()


# Dataset ----------------------------------------------------------------------
# Hyperparameters
# batch size=64
# number heads=4, hidden sizes=160, lr=0.001, gr_clip=0.1
from pytorch_forecasting.data import GroupNormalizer


max_prediction_length = 24
max_encoder_length = 7 * 24

# The dataset is split into training and validation sets
# training_cutoff is the latest timestamp in the dataset minus max_prediction_length
# Ensures that the last 24 hours are held out for validation
training_cutoff = time_df2["hours_from_start"].max() - max_prediction_length


training = TimeSeriesDataSet(
    time_df2[lambda x: x.hours_from_start <= training_cutoff],
    time_idx="hours_from_start",  # The time column is hours_from_start, which acts as the timeline.
    target="power_usage",  # The model is predicting power_usage, the main target variable.
    group_ids=["consumer_id"],  # The dataset contains multiple consumers, and the model will learn patterns separately for each consumer_id.
    min_encoder_length=max_encoder_length // 2,  # The model will use at least half of max_encoder_length (168/2 = 84 hours) of past data.
    max_encoder_length=max_encoder_length,  # The maximum past data used is 168 hours (7 days).
    min_prediction_length=1,  # The model must predict at least 1 future time step.
    max_prediction_length=max_prediction_length,  # The model can predict up to 24 time steps (hours) ahead.
    static_categoricals=["consumer_id"],  # consumer_id is a categorical variable that remains constant for each time series.
    time_varying_known_reals=[
        "hours_from_start",
        "day",
        "day_of_week",
        "month",
        "hour",
    ],  # These are known in advance and do not depend on predictions
    time_varying_unknown_reals=["power_usage"],
    target_normalizer=GroupNormalizer(
        groups=["consumer_id"], transformation="softplus"
    ),  # Normalizes power_usage values separately for each consumer_id group. #Uses SoftPlus transformation, which ensures positive values and smooths out variations.
    add_relative_time_idx=True,  # Adds a relative time index (helps the model generalize across different sequences).
    add_target_scales=True,  # Adds scaling parameters for power_usage, allowing the model to denormalize outputs.
    add_encoder_length=True,  # Adds the encoder sequence length as a feature, so the model knows how much past data is available.
)


# This function clones an existing TimeSeriesDataSet (in this case, training) but applies it to a new dataframe (time_df2).
# It ensures that the validation dataset has the same transformations, normalizations, and settings as the training dataset.
# This is the original full dataset (which includes both training and validation data).
# from_dataset will filter this dataframe appropriately based on the predict=True argument.
# predict(True) Purpose: It tells the function that this dataset will be used for prediction (not training).
# - The model will not use actual power_usage values as targets.
# - Instead, it will treat the power_usage column as unknown future values to be predicted.
validation = TimeSeriesDataSet.from_dataset(training, time_df2, predict=True, stop_randomization=True)


# create dataloaders for  our model ----------------------------------------------------------------------
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
# DataLoaders handle batching, shuffling, and loading data into memory during training.
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, persistent_workers=True)

# Baseline
import torch
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


# personal test ----------------------------------------------------------------------
# batch = next(iter(val_dataloader))  # Get the first batch
# x, (y, weight) = batch
# print("Input Features (x):", x)
# print("Target (y):", y)
# print("Weight:", weight)
# print("Actuals Device:", actuals.device)
# print("Baseline Predictions Device:", baseline_predictions.device)

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
logger = TensorBoardLogger("lightning_logs")

trainer = Trainer(
    max_epochs=45,
    accelerator="auto",
    devices="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=160,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
    optimizer="adam",
)

# train ----------------------------------------------------------------------
# trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# best_model_path = trainer.checkpoint_callback.best_model_path
# print(best_model_path)
# best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# print(best_tft)

# load saved best model
best_model_path = (
    "/Users/eungi/Desktop/eungi_dt/mim/livpop/src2/tft_practice/lightning_logs2/lightning_logs/version_3/checkpoints/epoch=5-step=2808.ckpt"
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

# average p50 loss per time series
print((actuals - predictions).abs().mean(axis=1))
# (1.0765 +  7.6859 +  2.4955 + 6.9401 + 14.6207)/5
# tensor([ 1.0765,  7.6859,  2.4955,  6.9401, 14.6207], device='mps:0')
# The last 2 time-series have a bit higher loss because their relative magnitude is also high.

# raw predict ----------------------------------------------------------------------
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
print(raw_predictions._fields)

print("\n")
print(raw_predictions.output._fields)
print(raw_predictions.output.prediction.shape)

# We get predictions of 5 time-series for 24 days[seems hours].
# For each day[seems hour] we get 7 predictions - these are the 7 quantiles:
# [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
# We are mostly interested in the 4th quantile which represents, let's say, the 'median loss'
# fyi, although docs use the term quantiles, the most accurate term are percentiles

# plot_prediction ----------------------------------------------------------------------
import matplotlib.pyplot as plt

for idx in range(5):
    print(idx)
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(
        raw_predictions.x,
        raw_predictions.output,
        idx=idx,
        add_loss_to_title=QuantileLoss(),
        ax=ax,
    )


# plot_prediction for other section of MT_004 ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
raw_prediction = best_tft.predict(
    training.filter(lambda x: (x.consumer_id == "MT_004") & (x.time_idx_first_prediction == 26512)),
    mode="raw",
    return_x=True,
)
best_tft.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0, ax=ax)


# out-of-sample pred ----------------------------------------------------------------------
# encoder data is the last lookback window: we get the last 1 week (168 datapoints) for all 5 consumers = 840 total datapoints

encoder_data = time_df[lambda x: x.hours_from_start > x.hours_from_start.max() - max_encoder_length]
last_data = time_df[lambda x: x.hours_from_start == x.hours_from_start.max()]

# decoder_data is the new dataframe for which we will create predictions.
# decoder_data df should be max_prediction_length*consumers = 24*5=120 datapoints long : 24 datapoints for each cosnumer
# we create it by repeating the last hourly observation of every consumer 24 times since we do not really have new test data
# and later we fix the columns

decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.Hour(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)

# fix the new columns ----------------------------------------------------------------------
decoder_data["hours_from_start"] = (decoder_data["date"] - earliest_time).dt.seconds / 3600 + (decoder_data["date"] - earliest_time).dt.days * 24
decoder_data["hours_from_start"] = decoder_data["hours_from_start"].astype("int")
decoder_data["hours_from_start"] += encoder_data["hours_from_start"].max() + 1 - decoder_data["hours_from_start"].min()

decoder_data["month"] = decoder_data["date"].dt.month.astype(np.int64)
decoder_data["hour"] = decoder_data["date"].dt.hour.astype(np.int64)
decoder_data["day"] = decoder_data["date"].dt.day.astype(np.int64)
decoder_data["day_of_week"] = decoder_data["date"].dt.dayofweek.astype(np.int64)

new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)


# create out-of-sample predictions for MT_002 ----------------------------------------------------------------------
new_prediction_data = new_prediction_data.query(" consumer_id == 'MT_002'")
new_raw_predictions = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
fig, ax = plt.subplots(figsize=(10, 5))
best_tft.plot_prediction(
    new_raw_predictions.x,
    new_raw_predictions.output,
    idx=0,
    show_future_observed=False,
    ax=ax,
)

# interpret ----------------------------------------------------------------------
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)

# Analysis on the training set ----------------------------------------------------------------------

predictions = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)


# create a new study ----------------------------------------------------------------------
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import pickle

# study = optimize_hyperparameters(
#    train_dataloader,
#    val_dataloader,
#    model_path="optuna_test",
#     n_trials=1,
#     max_epochs=1,
#     gradient_clip_val_range=(0.01, 1.0),
#     hidden_size_range=(30, 128),
#     hidden_continuous_size_range=(30, 128),
#     attention_head_size_range=(1, 4),
#     learning_rate_range=(0.001, 0.1),
#     dropout_range=(0.1, 0.3),
#     reduce_on_plateau_patience=4,
#     use_learning_rate_finder=False
# )

# save study results
# with open("test_study.pkl", "wb") as fout:
#    pickle.dump(study, fout)

abs_path = "/Users/eungi/Desktop/eungi_dt/mim/livpop/src2/tft_practice/test_study.pkl"
with open(abs_path, "rb") as fin:
    study2 = pickle.load(fin)

# print best hyperparameters
# print(study.best_trial.params)
print(study2.best_trial.params)

# ----------------------------------------------------------------------
type(decoder_data)
decoder_data.dtypes

z1 = time_df2.query("consumer_id == 'MT_002'")


z1 = 36000
z3 = 36003
# z3 += 32+1-z1
z3 = z3 + 32 + 1 - z1
z3

z1 = decoder_data["date"].dt

z2 = [last_data.assign(date=lambda x: x.date + pd.offsets.Hour(2))]
z2 = (decoder_data["date"] - earliest_time).dt
z3 = (decoder_data["date"] - earliest_time).dt.seconds / 3600
z4 = (decoder_data["date"] - earliest_time).dt.days * 24
z2.days * 24
z4
z5 = (decoder_data["date"] - earliest_time).dt.seconds / 3600 + (decoder_data["date"] - earliest_time).dt.days * 24


print(training)
print(training.data["target"])  # Might show the underlying DataFrame or a dictionary of processed values.
print(training.index)
val_dataloader
print(val_dataloader)
test = [y for x, (y, weight) in iter(val_dataloader)]
test[0]
actuals
baseline_predictions
