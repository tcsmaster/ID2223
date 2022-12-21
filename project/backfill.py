'''
import os
import modal


LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","pandas"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    '''
import hopsworks
import pandas as pd
import requests
import json


mode = hopsworks.login()
reg = mode.get_feature_store()
#api_url = "https://api.waqi.info/feed/@A189391/?token=67ad44804760b0db43223707670895e8324fd26e"
api_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/47.433,19.183/2020-12-25/2021-01-21?key=EA4A8Z2LC3D3KW5J96F4QFXFC"
response = requests.get(api_url)
print(json.dump(response.json(), indent = 4))
'''
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
'''