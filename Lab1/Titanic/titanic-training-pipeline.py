import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn", "xgboost"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        feature_view = fs.get_feature_view(name="titanic_modal", version=5)
    except:
        tit_fg = fs.get_feature_group(name="titanic_modal", version=5)
        query = tit_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal",
                                          version=5,
                                          description="Read from Titanic dataset",
                                          labels=["survived"],
                                          query=query)    

  
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)


    model = XGBClassifier(max_depth= 2, eta =  1, objective = 'binary:logistic')
    model.fit(X_train, y_train.values.ravel())


    y_pred = model.predict(X_test)

    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)


    df_cm = pd.DataFrame(results, ['True Passing', 'True Survival'],
                         ['Predicted Passing', 'Predicted Survival'])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    mr = project.get_model_registry()
    

    model_dir="titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    joblib.dump(model, model_dir + "/titanic_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")    



    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    tit_model = mr.python.create_model(
        name="titanic_modal", 
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Survivor Predictor"
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    tit_model.save(model_dir)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
