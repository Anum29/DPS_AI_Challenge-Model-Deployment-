
##  Overview
This repository contains code for deploying a machine learning model using Amazon SageMaker. The model is trained to predict accident values based on the year and month features. The deployment is done using a SageMaker endpoint, and the model is accessible for making predictions.

##  Contents
model_training.ipynb: Jupyter notebook for training and evaluating the machine learning model.
inference.py: Python script containing the inference code used by the SageMaker endpoint.
input.json: Sample JSON file containing input features for testing the deployed model.
requirements.txt: List of Python dependencies required for the inference environment.
model.tar.gz: Compressed archive containing the trained machine learning model.
out_scaler.sav: Scaler object for output features used during training.
scaler.sav: Scaler object for input features used during training.
Model Training
The model training is performed in the model_training.ipynb notebook. It uses a Random Forest Regressor from scikit-learn and is trained on the provided dataset. The trained model is then saved along with the scaler objects for input and output features.

## Inference
The inference.py script contains the code for making predictions using the deployed SageMaker endpoint. It loads the trained model and scaler objects, processes input data, and returns predictions.

## Deployment
The trained model is saved as a compressed archive (model.tar.gz) and uploaded to an S3 bucket.
The scaler objects (out_scaler.sav and scaler.sav) are also uploaded to the same S3 bucket.
The SageMaker endpoint is created using the SKLearnModel class, and the model, scaler objects, and inference code are configured.
The endpoint is deployed, and the model is ready to make predictions.
Testing the Model
The input.json file contains sample input data for testing the deployed model. The code snippet in the notebook demonstrates how to send a request to the SageMaker endpoint for inference.

## Dependencies
The requirements.txt file lists the Python dependencies required for running the inference code. These dependencies are automatically installed in the SageMaker environment during deployment.

## SageMaker Endpoint
The deployed SageMaker endpoint is named sagemaker-scikit-learn-2022-06-20-03-59-08-850. This endpoint can be used to make predictions by sending JSON payloads with input features.

Feel free to explore and use the provided code for deploying and testing machine learning models on SageMaker!
