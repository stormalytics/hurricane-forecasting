# Hurricane Forecasting

This project aims to enable the creation of hurricane forecasting products that are useful to decision-makers using deep learning methodologies. These forecasting products can include track (position) forecasts, intensity forecasts, wind forecasts, and storm surge forecasts. The goal is to achieve a relatively similar or smaller error than the official National Hurricane Center (NHC) forecast error.

## Published Work

Here are some previously published works in this area. Reading the background and methodology sections of these papers will give you more sources for other papers and datasets.

- Machine Learning in Tropical Cyclone Forecast Modeling: A Review
    - https://www.mdpi.com/2073-4433/11/7/676
    - Good review paper, start here
- Hurricane Forecasting: A Novel Multimodal Machine Learning Framework
    - https://arxiv.org/abs/2011.06125
    - Most recent major paper published on the topic, lots of reference to other previous paper in the background
    - Arguably has some flaws, but is promising in certain areas
    - Only 24-hour forecasts
- Fused Deep Learning for Hurricane Track Forecast from Reanalysis Data
    - https://hal.archives-ouvertes.fr/hal-01851001/
    - Example of how to combine track data and image data for forecasting
    - No comparison to official forecast error
- PHURIE: hurricane intensity estimation from infrared satellite imagery using machine learning
    - https://link.springer.com/article/10.1007/s00521-018-3874-6
    - Another cool example of ML and computer vision for forecasting
- Predicting Hurricane Trajectories Using a Recurrent Neural Network
    - https://arxiv.org/abs/1802.02548
    - Older paper with more basic work in this area
    - A combination of gridding the location data and using an RNN