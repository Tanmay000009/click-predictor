from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import logging
import joblib
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = joblib.load(open('vectorizer.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['GET'])
def home():
    return render_template("predict.html")


@app.route('/predict', methods=['POST'])
# def home():
#    return render_template("predict.html")
def predict():
        # for(j=0;j<)
        # int_features = [int(x) for x in request.form.values()]
        hour = request.form['hour']
        C1 = request.form['C1']
        banner_pos = request.form['banner_pos']
        site_category = request.form['site_category']
        app_category = request.form['app_category']
        device_type = request.form['device_type']
        device_conn_type = request.form['device_conn_type']
        C14 = request.form['C14']
        C15 = request.form['C15']
        C16 = request.form['C16']
        C17 = request.form['C17']
        C18 = request.form['C18']
        C19 = request.form['C19']
        C20 = request.form['C20']
        C21 = request.form['C21']

        int_features = [hour, C1, banner_pos, site_category, app_category, device_type, device_conn_type, C14,
                        C15, C16, C17, C18, C19, C20, C21]

        final_features = [np.array(int_features)]
        index_values = ['hour', 'C1', 'banner_pos','site_category', 'app_category', 'device_type', 'device_conn_type','C14','C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        df = pd.DataFrame([int_features], columns=  ['hour', 'C1', 'banner_pos','site_category', 'app_category', 'device_type', 'device_conn_type','C14','C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])
        final = list(df.T.to_dict().values())
        final = vectorizer.transform(final)
        prediction = model.predict(final)
        #
        # prediction *= 100
        # prediction = model.predict(final_features)
        # return prediction
        return render_template("predict.html", prediction_text='Click Output ${}'.format(prediction))




# .values()
# @app.route('/checker',methods=['POST','GET'])
# def checker():
#     if request.method == 'POST':
#         Expected_Answer = request.form['Expected_Answer']
#         Students_Answer = request.form['Students_Answer']
#         print(Expected_Answer)
#         print(Students_Answer)
#         match = [Expected_Answer,Students_Answer]
#         vectorizer.bert(match)
#         vectors_bert = vectorizer.vectors
#
#         dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])
#         score = (1 - dist_1) * 100
#         print((1 - dist_1) * 100)
#         print('dist_1: {0}'.format(dist_1))
#         return render_template("checker.html",Students_Answer=Students_Answer,Expected_Answer=Expected_Answer,score=score)
#     return  render_template("checker.html")

if __name__ == '__main__':
    app.run()
