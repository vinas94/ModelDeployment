import json
import pickle
import numpy as np
import sklearn
from flask import Flask, request
from mysql_con import push_to_sql

app = Flask(__name__)

@app.route('/')
def hello():
    """ Main page of the app. """
    return "Hello! Let me classify your beanss!"

@app.route('/predict', methods=['POST'])
def predict():
    """ Return JSON serializable output from the model """
    payload = request.json
    X = np.array(json.loads(payload['data']))
    with open('DryBeanSVM', 'rb') as file:
        classifier = pickle.load(file)
    predictions = classifier.predict(X)
    push_to_sql(predictions)

    return {'predictions': json.dumps(predictions.tolist())}


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)