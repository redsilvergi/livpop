#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:20:09 2025

@author: eungi
"""
# # libraries
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import sklearn
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, r2_score, mean_absolute_error
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.datasets import make_regression
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import RepeatedKFold, cross_val_score
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split, GridSearchCV
# import optuna
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.metrics import mean_absolute_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input
# from sklearn.metrics import confusion_matrix, classification_report

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV
# from xgboost import XGBClassifier
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, accuracy_score
# import shap
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from catboost import CatBoostClassifier
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, StratifiedKFold

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split

# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# import matplotlib.pyplot as plt

# import optuna
# from xgboost import XGBRegressor
# from tensorflow.keras.layers import Dense, Dropout, Input
# import sklearn
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, r2_score, mean_absolute_error
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.datasets import make_regression
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import RepeatedKFold, cross_val_score

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# from catboost import Pool
# from sklearn.ensemble import VotingClassifier

# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from catboost import CatBoostClassifier
# from sklearn.metrics import accuracy_score, f1_score
# import optuna
# import joblib


import pandas as pd

df_tft = pd.read_csv(
    "/Users/eungi/Desktop/eungi_dt/mim/livpop/src2/data/df_tft_saved.csv",
    encoding="UTF-8",
)

# # 2007-1 ~ 2023-12 컬럼만 추출
# time_cols = df.columns[5:209]  # 6~209열 (인덱스는 0부터 시작)

# # 연도별로 NaN이 없는 행 개수 저장할 딕셔너리
# non_nan_counts = {}

# # 2007~2023까지 순차적으로 연도 범위를 조정하면서 NaN이 없는 행 개수 확인
# for start_year in range(2007, 2024):
#     # 해당 연도부터 2023년까지의 컬럼 선택
#     selected_cols = [col for col in time_cols if int(col.split("-")[0]) >= start_year]

#     # 선택된 열들에서 NaN이 없는 행 개수 계산
#     non_nan_rows = df[selected_cols].dropna().shape[0]

#     # 결과 저장
#     non_nan_counts[f"{start_year}~2023"] = non_nan_rows

# # 결과를 DataFrame으로 변환
# result_df = pd.DataFrame(list(non_nan_counts.items()), columns=["연도 범위", "NaN 없는 행 개수"])

# # 결과 출력
# print(result_df)


# # 데이터 long format으로 변환
# df_melted = df.melt(id_vars=["지점번호", "호선", "구간명", "차로수", "주소"], var_name="날짜", value_name="교통량")

# # 날짜 컬럼을 datetime 형식으로 변환
# df_melted["날짜"] = pd.to_datetime(df_melted["날짜"], format="%Y-%m")

# # 2013년부터 2023년까지의 데이터만 필터링
# df_filtered = df_melted[(df_melted["날짜"] >= "2013-01") & (df_melted["날짜"] <= "2023-12")]

# # 각 지점별로 2013~2023 데이터 개수 확인
# valid_stations = df_filtered.groupby("지점번호")["교통량"].count()

# # 2013~2023까지 모든 데이터(11년 x 12개월 = 132개)가 존재하는 지점만 선택
# valid_stations = valid_stations[valid_stations == 132].index

# # 해당 지점의 데이터만 필터링
# df_clean = df_filtered[df_filtered["지점번호"].isin(valid_stations)]

# # 결과 확인
# print(f"결측값 없이 2013~2023 데이터를 포함하는 지점 개수: {len(valid_stations)}")
# df_clean.head()


# # 필요한 컬럼만 선택
# df_tft = df_clean[["지점번호", "날짜", "교통량"]].copy()

# # '날짜'를 숫자로 변환 (TFT는 datetime을 그대로 받지 않음)
# df_tft["날짜"] = df_tft["날짜"].astype("int64") // 10**9  # Unix timestamp 변환

# # 정규화: 지점별로 교통량을 정규화할 수 있음
# df_tft["교통량"] = df_tft.groupby("지점번호")["교통량"].transform(lambda x: (x - x.mean()) / x.std())

# # 확인
# df_tft.head()

# # '날짜'를 연속된 time_idx로 변환
# df_tft = df_clean[["지점번호", "날짜", "교통량"]].copy()

# # '날짜'를 datetime으로 변환
# df_tft["날짜"] = pd.to_datetime(df_tft["날짜"], format="%Y-%m")

# # 최소 날짜를 기준으로 time_idx 생성 (연속 숫자)
# df_tft["time_idx"] = (df_tft["날짜"].dt.year - df_tft["날짜"].dt.year.min()) * 12 + df_tft["날짜"].dt.month

# # 정규화: 지점별로 교통량을 정규화할 수 있음
# df_tft["교통량"] = df_tft.groupby("지점번호")["교통량"].transform(lambda x: (x - x.mean()) / x.std())

# # 확인
# df_tft.head()

import torch
from lightning.pytorch import Trainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# 설정
max_encoder_length = 24  # 과거 24개월을 입력으로 사용
max_prediction_length = 12  # 미래 12개월을 예측
batch_size = 64  # 배치 크기

training_cutoff = df_tft["time_idx"].max() - max_prediction_length

# 데이터셋 생성
training = TimeSeriesDataSet(
    df_tft[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="교통량",
    group_ids=["지점번호"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["지점번호"],  # 측정 지점 ID (고정된 값)
    time_varying_known_reals=["time_idx"],  # 미래에도 아는 값 (시계열 인덱스)
    time_varying_unknown_reals=["교통량"],  # 예측할 값
    target_normalizer=GroupNormalizer(groups=["지점번호"]),
)

validation = TimeSeriesDataSet.from_dataset(training, df_tft, predict=True, stop_randomization=True)

# 데이터 로더 생성
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10, persistent_workers=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, persistent_workers=True)


from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

# 모델 생성
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
    optimizer="adam",
)

# PyTorch Lightning Trainer 설정
trainer = Trainer(
    max_epochs=12,
    accelerator="auto",
    devices="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

# 모델 학습
# train ----------------------------------------------------------------------
# trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# best_model_path = trainer.checkpoint_callback.best_model_path
# print(best_model_path)
best_model_path = "/Users/eungi/tmp/lightning_logs/lightning_logs/version_0/checkpoints/epoch=11-step=1320.ckpt"
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
print(best_tft)


# predict ----------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available, else CPU
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

for idx in range(30):
    print(idx)
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(
        raw_predictions.x,
        raw_predictions.output,
        idx=idx,
        add_loss_to_title=QuantileLoss(),
        ax=ax,
    )
