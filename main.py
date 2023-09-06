from flask import Flask

app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def custom_check():
    return "Hello world"


@app.route("/predict", methods=["POST"])
def predict_genre():
    return "Index Page"


@app.route("/train")
def hello():
    return "Hello, World"
