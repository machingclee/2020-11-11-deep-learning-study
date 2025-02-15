import requests
URL = "http://localhost:80/predict"

TEST_AUDIO_FILE = "./test/go.wav"


if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE, "rb")
    values = {"file": (TEST_AUDIO_FILE, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("Predicted Keyword is: {}".format(data["keyword"]))
