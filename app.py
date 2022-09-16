from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import joblib
import pickle

app = Flask(__name__)
model = joblib.load('regressor.pkl')


@app.route('/')
@app.route('/main')
def main():
    return render_template("heartpred.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [[x for x in request.form.values()]]
    c = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = pd.DataFrame(int_features, columns=c)
    semires = model.predict(df)
    if semires == 0:
        result = "No Heart Problem"
    else:
        result = "Heart Problem"

    return render_template("heartpred.html", prediction_text=" Patient has : {}".format(result))


if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=7000)
