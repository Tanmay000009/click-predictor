from flask import Flask, request,render_template
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
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
        df = pd.DataFrame([int_features], columns=  ['hour', 'C1', 'banner_pos','site_category', 'app_category', 'device_type', 'device_conn_type','C14','C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])
        final = list(df.T.to_dict().values())
        final = vectorizer.transform(final)
        prediction1 = model.predict_proba(final)
        prediction1 *= 100
        prediction1 = prediction1[0][1]
        prediction1 = round(prediction1,3)
        return render_template("predict.html", prediction_text='Probablity of clicking the add is {} %'.format(prediction1))

if __name__ == '__main__':
    app.run()
