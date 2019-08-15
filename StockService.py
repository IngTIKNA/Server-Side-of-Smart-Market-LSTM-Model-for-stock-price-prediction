from flask import Flask, jsonify
import numpy as np
from decimal import Decimal                         
from keras.models import load_model
import pandas_datareader as pdr
import time
app = Flask(__name__)

@app.route("/")
def hello():
    dfNew = pdr.get_data_yahoo('TSLA', start="2019-7-1")
    datasetNew = np.array(dfNew[["Close"]])
    res = np.around(datasetNew[-1], decimals=2)
    return jsonify({"CurrentValue" : str(res)})

while(True):
    time.sleep(2)
    app.run(debug=True, host="0.0.0.0")