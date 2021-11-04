import json
import pickle
import sklearn
import numpy as np
from flask import Flask, request, render_template

# Create a Flask app object
app = Flask(__name__)

@app.route('/')
def hello():
    ''' Main page of the app '''
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' Return JSON serializable output from the model '''

    # Capture the payload
    payload = request.json
    X = np.array(json.loads(payload['data']))
    store_preds = True
    if 'store' in payload:
        store_preds = payload['store']

    # Load the classifier
    with open('DryBeanSVM', 'rb') as file:
        classifier = pickle.load(file)

    # Get the predictions
    predictions = classifier.predict(X)

    # Store them in the DB
    if store_preds:
        from mysql_con import push_to_sql
        push_to_sql(predictions)

    return {'predictions': json.dumps(predictions.tolist())}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)