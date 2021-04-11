from logging import debug
from flask import Flask, request, jsonify
import os
from keyword_spotting_service import Keyword_Spotting_Service
import random
app = Flask(__name__)


@app.route("/", methods=["GET"])
def get_version():
    return jsonify({"version": 0.01})


@app.route("/predict", methods=["POST"])
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 10000000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = Keyword_Spotting_Service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted keyword in json
    data = {"keyword": predicted_keyword}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=3800)
    print("service is listening at port 3800")
