# Lab 1
In the first lab I set up my first serverless machine learning system.

The first machine learning problem is the classical Iris dataset prediction. I stored the features in my Hopsworks FeatureStore, trained the serverless K-nearest neighbours algorithm using Modal, and hosted a Gradio application on Huggingface Spaces.

For the second part, I created a similar system, but for a different famous dataset, the Titanic dataset. Since there are a lot of missing values, I created a separate Jupyter Notebook for EDA. Eventually I chose four features: age, sex, ticket class and fare.

In titanic-feature-pipeline.py, I read the titanic data, drop irrelevant features, clean up missing values, encode the Sex parameter, and write them to a Feature Group.

In titanic-training-pipeline.py, I create a Feature View of the Group, and train an XGBoost algorithm with mostly default features. Then I store this model in Hopsworks.

In titanic-feature-pipeline-daily.py, I wrote a function that creates a synthetic person, either a survivor or someone who passed away. The data to generate these people comes from EDA analysis (or historical knowledge) that adult first-class women were the likely to survive, and third-class adult men were the least likely to stay alive.

In titanic-batch-inference-pipeline.py, I predict whether the synthetic passenger would have survived the Titanic or not.

There are two additional Huggingface spaces, one where one can enter different parameters, and the model predict whether the person ould have died or not.
The second space serves as a dashboard where the most recent daily pediction can be seen, as well as the last 5 predictions plus the confusion matrix of the predictions
