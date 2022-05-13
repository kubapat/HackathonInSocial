import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

from main import predict, removeUnnecessaryChars, getQuestionList

app = Flask(__name__)

@app.route("/", methods=['GET'])
def bootstrap():
    return render_template('index.html', question='Question?', indicator='indicator: POSITIVE/NEGATIVE')


@app.route("/api/get-follow-up-question", methods=['POST'])
def followUpQuestion():
    json_ = request.get_json()

    string = removeUnnecessaryChars(json_['text'])
    modelVectorizer = joblib.load('output/model_vectorizer.pkl')
    modelLr = joblib.load('output/model_lr.pkl')

    result = modelLr.predict(modelVectorizer.transform([string]))
    return jsonify({
        'result': result.tolist(),
        'questions': getQuestionList(string)
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
