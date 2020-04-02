# Disaster Response Pipeline Project
In this project, we will work with disaster data obtained from [Figure Eight](https://www.figure-eight.com).
This project includes three parts:
1) a ETL pipeline for data cleaning and preprocessing
2) a Machine Learning Pipeline to train a model based on current collected data. The model will need to classify new messages received into different categories. 
2) a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
