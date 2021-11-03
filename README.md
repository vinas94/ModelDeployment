# Deploying a Machine Learning Model
### This is a step-by-step walkthrough of creating a simple classification model and deploying it via Google Kubernetes Engine (GKE) using Flask and Docker.

<img src="./flask_docker_k8.png">

<br>

## Contents

The project is split into two key parts - (1) local development which sets up the model and exposes an API on a local network and (2) deployment of said model on GKE.

**The following will be covered in order of occurance:**
- Creating a simple classification model with scikit-learn
- Creating a Flask app with the said model
- Connecting a MySQL database to store the results
- Packaging both the Flask app and the MySQL instance into Docker containers
- Combining both containers via Docker-Compose
- Running these containers locally

At this stage we have a working app. Next up is to host it somewhere so that it can be accessed from anywhere at all times. For such a simple app, Kubernetes is more than an overkill, better alternatives exist. However, this walkthrough is not about being pragmatic, so it will continue by:

- Enabling GKE and Artifact Registry on Google Cloud Platform
- Uploading the app and its dependancies to the Artifact Registry
- Converting Docker-Compose file to Kubernetes Resources
- Launching the app

<br>

## Local development
### 1. Creating a classification model

For the purposes of this walkthrough a simple classification model will be built using the scikit-learn library. [The Dry Beans dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) from the UCI repository will be used for training the model. The dataset consists of a single target feature (one of seven dry bean types) and 16 numerical feature variables defining bean characteristics such as size, shape and roundness. There are in total 13611 observations. A Support Vector Clasifier will be used to classify these beans as it is quick to fit and often gives good results out of the box.

```
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Loading the data and separating target from the features
df = pd.read_csv('./data/Dry_Bean_Dataset.csv', encoding='unicode_escape')
y = df['Class']
X = df.drop('Class', axis=1)

# Splitting the data to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Fitting a Support Vector Classifier with a linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
preds = svm.predict(X_test)
accuracy = (preds==y_test).mean()

print(f'Prediction accuracy on test data: {"%.3f"%accuracy}')
>> Prediction accuracy on test data: 0.923
```

The fitted model provides a 92% accuracy on the test data. However, it is hard to say if this is a good result in absence of a baseline or without knowing more about the dataset. For our purposes though, this will do, therefore it is time to store the model in a file as such:

```
with open('./app/DryBeanSVM', 'wb') as file:
    pickle.dump(svm, file)
```

<br>

### 2. Creating a Flask app

The newly constructed model now needs to be wrapped within an app so that it could be simply used via an API call. Flask makes this straighforward. Below is an implementation for a DryBeans classification app.

```
import json
import pickle
import numpy as np
import sklearn
from flask import Flask, request, render_template
from mysql_con import push_to_sql

# Create a Flask app object
app = Flask(__name__)

@app.route('/')
def hello():
    """ Main page of the app. """
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Return JSON serializable output from the model """

    # Capture the payload
    payload = request.json
    X = np.array(json.loads(payload['data']))

    # Load the classifier
    with open('DryBeanSVM', 'rb') as file:
        classifier = pickle.load(file)

    # Get the predictions
    predictions = classifier.predict(X)

    # Store them in the DB
    push_to_sql(predictions)

    return {'predictions': json.dumps(predictions.tolist())}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

It has two access points. One is a root `'/'` access which leads to a home page that returns a static output defined by `/app/templates/home.html`. For the sake of simplicity, the `hello()` function could be changed to return any string and the functionality of the app would stay the same.

The second access point is the `'/predict'` route which expects a json payload containing all 16 features of one or more beans. The payload is parsed and sent to the model which itself is loaded from a pickle file created earlier. Predictions are then returned as a json string as well as stored in a database (more on this in the next section).

<br>

### 3. Connecting to a database

The Flask app defined earlier called a `push_to_sql` function which was imported from a separate script. That script is defined below. Its purpose is to establish a connection with an existing instance of a MySQL database.

```
import os
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Create the MySQL connection string
# These environment variables will be passed to the system during container startup
db_string = f"mysql://{os.environ['MYSQL_USER']}: \
                      {os.environ['MYSQL_PASSWORD']}@ \
                      {os.environ['MYSQL_HOST']}:3306/ \
                      {os.environ['MYSQL_DB']}"

# Establish the connection
db = create_engine(db_string)
base = declarative_base()
session = sessionmaker(db)()

# Create a table for storing the predictions if it doesn't exist yet
db.execute('''
    CREATE TABLE IF NOT EXISTS drybeans (
	id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')

class DryBeansSQL(base): 
    ''' Defining new entries '''
    __tablename__ = 'drybeans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    name = Column(String)

def push_to_sql(array):
    '''  Pushing new data to storage '''
    for i in array:
        drybean = DryBeansSQL(name=i)
        session.add(drybean) 
        
    session.commit()
    print('Data upload complete!')
```


<br>

### 4. Creating Docker containers


```
pipreqs .
```

```
# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y default-libmysqlclient-dev gcc
RUN pip3 install mysqlclient==2.0.3
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
```

```
docker build -t vinas94/drybeans-img .
```


<br>

### 5. Defining a Docker-Compose file


```
version: '3.7'

services:
  app:
    image: vinas94/drybeans-img
    ports:
    - 5000:5000
    working_dir: /app
    environment:
      MYSQL_HOST: mysql
      MYSQL_USER: root
      MYSQL_PASSWORD: secret
      MYSQL_DB: drybeans_db

  mysql:
    image: mysql:5.7
    ports:
    - 3308:3306
    volumes:
    - drybeans-vol:/var/lib/minesql
    environment:
      MYSQL_ROOT_PASSWORD: secret
      MYSQL_DATABASE: drybeans_db

volumes:
  drybeans-vol:
```


<br>

### 6. Launching the app


```
docker-compose up
```

