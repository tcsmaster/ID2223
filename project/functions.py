import requests
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)


def get_air_quality_data(station_name:str) -> list:
    AIR_QUALITY_API_KEY = os.getenv('AIR_QUALITY_API_KEY')
    request_value = f'https://api.waqi.info/feed/{station_name}/?token={AIR_QUALITY_API_KEY}'
    answer = requests.get(request_value).json()["data"]
    forecast = answer['forecast']['daily']
    return [
        answer["time"]["s"][:10],      # Date
        int(forecast['pm25'][0]['avg']),  # avg predicted pm25
        int(forecast['pm10'][0]['avg']),  # avg predicted pm10
        max(int(forecast['pm25'][0]['avg']), int(forecast['pm10'][0]['avg'])) # avg predicted aqi
    ]

def get_air_quality_df(data:list)-> pd.DataFrame:
    col_names = [
        'date',
        'pm25',
        'pm10',
        'aqi'
    ]

    new_data = pd.DataFrame(
        data
    ).T
    new_data.columns = col_names
    new_data['pm25'] = pd.to_numeric(new_data['pm25'])
    new_data['pm10'] = pd.to_numeric(new_data['pm10'])
    new_data['aqi'] = pd.to_numeric(new_data['aqi'])

    return new_data


def get_weather_data_daily(city: str) -> list:
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    answer = requests.get(f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=json').json()
    data = answer['days'][0]
    return [
        answer['address'].lower(),
        data['datetime'],
        data['tempmax'],
        data['tempmin'],
        data['temp'],
        data['feelslikemax'],
        data['feelslikemin'],
        data['feelslike'],
        data['dew'],
        data['humidity'],
        data['precip'],
        data['precipprob'],
        data['precipcover'],
        data['snow'],
        data['snowdepth'],
        data['windgust'],
        data['windspeed'],
        data['winddir'],
        data['pressure'],
        data['cloudcover'],
        data['visibility'],
        data['solarradiation'],
        data['solarenergy'],
        data['uvindex'],
        data['conditions']
    ]

def get_weather_df(data:list)->pd.DataFrame:
    col_names = [
        'name',
        'date',
        'tempmax',
        'tempmin',
        'temp',
        'feelslikemax',
        'feelslikemin',
        'feelslike',
        'dew',
        'humidity',
        'precip',
        'precipprob',
        'precipcover',
        'snow',
        'snowdepth',
        'windgust',
        'windspeed',
        'winddir',
        'pressure',
        'cloudcover',
        'visibility',
        'solarradiation',
        'solarenergy',
        'uvindex',
        'conditions'
    ]

    new_data = pd.DataFrame(
        data
    ).T
    new_data.columns = col_names
    for col in col_names:
        if col not in ['name', 'date', 'conditions']:
            new_data[col] = pd.to_numeric(new_data[col])
            if col in ['uvindex', 'precipprob']:
                new_data[col] = new_data[col].astype('int64')

    return new_data

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["windgust"].isna(),'windgust'] = df['windspeed']
    df['snow'].fillna(0,inplace=True)
    df['snowdepth'].fillna(0, inplace=True)
    if "sealevelpressure" or "datetime" in df.columns:
        df.rename(columns={"sealevelpressure":"pressure", "datetime":"date"}, inplace=True)
    df['pressure'].fillna(df['pressure'].mean(), inplace=True)
    return df

def data_encoder(X):
    from sklearn.preprocessing import OrdinalEncoder
    X.drop(columns=['date', 'name'], inplace=True)
    X['conditions'] = OrdinalEncoder().fit_transform(X[['conditions']])
    return X