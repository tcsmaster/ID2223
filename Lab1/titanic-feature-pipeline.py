import os
import numpy as np
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    
    titanic_df['Age'].fillna(np.random.randint(np.floor(titanic_df['Age'].min()), np.ceil(titanic_df['Age'].max())), inplace = True)
    
    titanic_df['Sex'].replace({'female': 0, 'male' : 1}, inplace=True)

    titanic_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

    tit_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=5,
        primary_key=['Fare'],
        description="Titanic dataset")
    tit_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
