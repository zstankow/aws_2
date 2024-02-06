from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello():
    print('Hello world!')
    return 'hello'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
