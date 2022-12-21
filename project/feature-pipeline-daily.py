import os
import modal
from functions import *

LOCAL = True

if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks", "requests"]) 

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def g():
    import requests
    import hopsworks

    client = hopsworks.login()
    seto = client.get_feature_store()


    station_name = "A189391"
    qual_data = get_air_quality_data(station_name)
    daily_qual_df = get_air_quality_df(qual_data)

    long, lat = 47.437,	19.256
    daily_weather_df = get_weather_df(get_weather_data(long, lat))

    


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()