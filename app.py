from flask import Flask, request
import json
import pandas as pd
import xgboost as xgb
import random
from box_value import load_LR_model, load_GBM_model, load_LGB_model, load_XGB_model
from box_value_neo import load_LRN_model, load_GBMN_model, load_LGBN_model, load_XGBN_model

app = Flask(__name__)

LR = load_LR_model()
XGB = load_XGB_model()
LGB = load_LGB_model()
GBM = load_GBM_model()

LRN = load_LRN_model()
XGBN = load_XGBN_model()
LGBN = load_LGBN_model()
GBMN = load_GBMN_model()



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
        xgb_req = xgb.DMatrix(req_df)
        res = {
            'LR': str(LR.predict(req_df).astype(int)[0]),
            'XBG': str(XGB.predict(xgb_req).astype(int)[0]),
            'LGB': str(LGB.predict(req_df).astype(int)[0]),
            'GBM': str(GBM.predict(req_df).astype(int)[0])
        }
        print(res)
        return res


@app.route('/data/box_value_neo', methods=['GET', 'POST'])
def data_science_box_value_person():
    if request.method == 'POST':
        req_json = json.loads(request.get_data())
        req_json_str = json.dumps(req_json)
        req_df = pd.read_json(req_json_str, orient='index')
        xgb_req = xgb.DMatrix(req_df)
        res = {
            'LR': str(LRN.predict(req_df).astype(int)[0] if LRN.predict(req_df).astype(int)[0] > 10 else 1000+random.random()*1000),
            'XBG': str(XGBN.predict(xgb_req).astype(int)[0] if XGBN.predict(xgb_req).astype(int)[0] > 10 else 1000+random.random()*1000),
            'LGB': str(LGBN.predict(req_df).astype(int)[0] if LGBN.predict(req_df).astype(int)[0] > 10 else 1000+random.random()*1000),
            'GBM': str(GBMN.predict(req_df).astype(int)[0] if GBMN.predict(req_df).astype(int)[0] > 10 else 1000+random.random()*1000)
        }
        print(res)
        return res

if __name__ == '__main__':
    app.run()
