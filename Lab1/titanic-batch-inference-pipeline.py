import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=5)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    offset = 3
    data= y_pred[y_pred.size-offset]
    
    if data == 0:
        prediction = "This person was predicted not to survive the Titanic."
        pred = "Deceased"
    else:
        prediction = "This person was predicted to  survive the Titanic."
        pred = "Survivor"

    with open('latest_prediction.txt', 'w') as f:
        f.write(prediction)
    
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_prediction.txt", "Resources/predictions", overwrite=True)
    
    iris_fg = fs.get_feature_group(name="titanic_modal", version=5)
    df = iris_fg.read()
    label = df.iloc[-offset]["survived"]

    if label == 0:
        prediction = "This person did not survive the Titanic."
        data2 = "Deceased"
    else:
        prediction = "This person probably survived the Titanic."
        data2 = "Survivor"

    with open('actual_prediction.txt', 'w') as f:
        f.write(prediction)
    
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./actual_prediction.txt", "Resources/predictions", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Prediction Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [pred],
        'label': [data2],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our feature group has examples of all 2 different outcomes:
    print("Number of different survivor predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Passing', 'True Survival'],
                             ['Predicted Passing', 'Predicted Survival'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need both a deceased and a survivor prediction to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get both predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()