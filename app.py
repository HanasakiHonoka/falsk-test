from flask import Flask, request
import json
import pandas as pd
from LR_model import load_LR_model
from LR_person_model import load_LR_person_model

app = Flask(__name__)

LR = load_LR_model()
LR_person = load_LR_person_model()
# gunicorn -w 4 -b 106.54.68.249:10012 app:app
# gunicorn -w 2 -b 0.0.0.0:10012 app:app

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/data/box_value', methods=['GET', 'POST'])
def data_science_box_value():
    if request.method == 'POST':
        req_json = json.loads(request.get_data())
        req_json_str = json.dumps(req_json)
        req_df = pd.read_json(req_json_str, orient='index')
        print(LR.predict(req_df).astype(int))
        return str(LR.predict(req_df).astype(int)[0])


@app.route('/data/box_value_person', methods=['GET', 'POST'])
def data_science_box_value_person():
    if request.method == 'POST':
        req_json = json.loads(request.get_data())
        req_json_str = json.dumps(req_json)
        req_df = pd.read_json(req_json_str, orient='index')
        print(LR_person.predict(req_df).astype(int))
        return str(LR_person.predict(req_df).astype(int)[0])

if __name__ == '__main__':
    app.run()
