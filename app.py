from flask import Flask, render_template, request
import os
import sys

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictmarks', methods=['GET', 'POST'])
def predictmarks():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        logging.info(pred_df)
        logging.info("Staring Prediction")

        predict_pipeline=PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        logging.info("Prediction Completed")

        return render_template('home.html', results=result[0])
        


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)