'''
import os
import modal


LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

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
api_url = "https://api.waqi.info/feed/geo:47.438;19.026/?token=67ad44804760b0db43223707670895e8324fd26e"
response = requests.get(api_url)
print(json.dumps(response.json(), indent=4))
'''
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
'''