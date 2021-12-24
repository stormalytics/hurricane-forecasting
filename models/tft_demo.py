import copy
from pathlib import Path
import warnings
from pprint import pprint as pp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


hurdat_table = pd.read_csv('./data/hurdat/hurdat2_processed.csv')
total_size = len(hurdat_table)

hurdat_table_train = hurdat_table[hurdat_table["year"] < 2015]
hurdat_table_test = hurdat_table[hurdat_table["year"] >= 2015]

encoder_length = 12
prediction_length = 12

train_dataset = TimeSeriesDataSet(
    hurdat_table_train,
    time_idx="time_idx",
    group_ids=["atcf_code"],
    target=["longitude", "latitude"],
    min_encoder_length=encoder_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=encoder_length,
    min_prediction_length=prediction_length,
    max_prediction_length=prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["hour", "jday"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall",
                                "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth",
                                "x", "y", "vpre"],
    add_relative_time_idx=True
)

test_dataset = TimeSeriesDataSet(
    hurdat_table_test,
    time_idx="time_idx",
    group_ids=["atcf_code"],
    target=["longitude", "latitude"],
    min_encoder_length=encoder_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=encoder_length,
    min_prediction_length=prediction_length,
    max_prediction_length=prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["hour", "jday"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall",
                                "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth",
                                "x", "y", "vpre"],
    add_relative_time_idx=True
)

BATCH_SZIE = 128

train_dataloader = train_dataset.to_dataloader(batch_size=BATCH_SZIE)
test_dataloader = test_dataset.to_dataloader(batch_size=BATCH_SZIE)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning ratecon
logger = TensorBoardLogger("./models/tft_logging", name="tft_demo")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=30,
    # max_epochs=1,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=[7,7],  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)

# trainer.fit(
#     tft,
#     train_dataloader=train_dataloader,
#     val_dataloaders=test_dataloader,
# )


# best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint("./models/tft_logging/tft_demo/version_1/checkpoints/epoch=29-step=899.ckpt")


actuals = torch.concat([torch.stack(y[0], dim=-1) for x, y in iter(test_dataloader)], dim=0)
predictions = torch.stack(tft.predict(test_dataloader), dim=-1)

print(actuals[0])
print(predictions[0])
# print(predictions.shape)


def haversine(pred, actual, batch=True):
    R = 6371

    pred = pred.reshape((-1, prediction_length, 2))
    actual = actual.reshape((-1, prediction_length, 2))

    if batch:
        lon_actual, lon_pred = actual[:, :, 0], pred[:, :, 0]
        lat_actual, lat_pred = actual[:, :, 1], pred[:, :, 1]
    else:
        lon_actual, lon_pred = actual[:, 0], pred[:, 0]
        lat_actual, lat_pred = actual[:, 1], pred[:, 1]

    lon_actual, lon_pred = np.deg2rad(lon_actual), np.deg2rad(lon_pred)
    lat_actual, lat_pred = np.deg2rad(lat_actual), np.deg2rad(lat_pred)

    alpha = np.sin((lat_pred-lat_actual)/2)**2 + np.cos(lat_pred) * \
        np.cos(lat_actual)*np.sin((lon_pred-lon_actual)/2)**2
    d = 2*R*np.arcsin(np.sqrt(alpha))

    d = d*0.539957

    return d


# error_distances = haversine(predictions, actuals)
# print(error_distances.shape)

# error_distances_df = pd.DataFrame(error_distances.numpy().astype("float"))
# columns = [f'future_horizon_{(i+1)*6}' for i in range(prediction_length)]
# error_distances_df.columns = columns
# print(error_distances_df.describe())
# error_distances_df.to_csv("./results/track_error_tft.csv", index=False)

# sns.displot(data=error_distances_df, kind="kde")
# plt.xlim(0, 400)
# plt.gca().set_xlabel('Forcast Error (n mi)')
# plt.show()



