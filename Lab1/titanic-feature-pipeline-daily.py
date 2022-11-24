import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image", "xgboost"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_person(pclass, sex, age_min, age_max, fare_min, fare_max):
    """
    Returns a single person as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "Age": [float(random.randrange(age_min, age_max))],
                       "Sex": sex,
                       "Fare": [random.uniform(fare_min, fare_max)],
                       "Pclass": pclass
                      })
    df["Survived"] = 1 - sex
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    survivor = generate_person(1, 0, 2, 80, 2.2, 300)
    deceased = generate_person(3, 1, 2, 80, 2.2, 300)

    pick_random = random.uniform(0,2)

    if pick_random >= 1:
        person = survivor
        print("Survivor added")
    else:
        person = deceased
        print("Deceased added")

    return person

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        titanic_df = get_random_passenger()

    tit_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=5,
        primary_key=["Fare"], 
        description="Titanic dataset")
    tit_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
