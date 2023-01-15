# Project

In this project I'm going to implement an air quality monitoring and prediction application for the city of Vienna, that is, to predict th Air Quality Index (AQI) from weather data attributes, like snowcover, pressure, temperature, etc.

The project consisted of 4 parts:

- backfill historical data into my Hopsworks Feature Store
- a serverless feature pipeline hich writes new data to the Feature Store evey day
- a training pipeline, where I trained an XGBoost model.
- a Huggingface Space, where I deploy the model to predict AQI from weather pedictions for the next 7 days.

The website is available [here](https://huggingface.co/spaces/CsanadT/Air_Quality_Index)

## Backfill

The code can be found in the notebooks. The AQI data comes from [The  World Air Quality Index](https://aqicn.org/data-platform/register/) database, the weather data is available at [Visual Crossing](https://www.visualcrossing.com/) database. The historical data dates back to 2014, providing sufficient training data. Unfortunately for AQI, data is not available through their API, only downloading as a csv file. For weather data, my account only allowed for 1000 record in 24 hours.
