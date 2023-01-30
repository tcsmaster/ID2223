from functions import *
import hopsworks

client = hopsworks.login()
fs = client.get_feature_store()

station_name = "vienna"

daily_qual_data = get_air_quality_data(station_name)
daily_qual_df = get_air_quality_df(daily_qual_data)

daily_weather_data = get_weather_data_daily(station_name)
raw_daily_weather_df = get_weather_df(daily_weather_data)
daily_weather_df = transform(raw_daily_weather_df)

air_quality_fg = fs.get_or_create_feature_group(
    name = 'air_quality_fg',
    version = 3
)
weather_fg = fs.get_or_create_feature_group(
    name = 'weather_fg',
    version = 5
)

air_quality_fg.insert(daily_qual_df)
weather_fg.insert(daily_weather_df)
