import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all the sub dirs:
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        if dir_path is not dataset_path:
            category_name = dir_path.split(os.path.sep)[-1]
            print(f"Processsing {category_name}")
            data["mappings"].append(category_name)

            for f in file_names:
                file_path = os.path.sep.join([dir_path, f])

                signal, sample_rate = librosa.load(file_path)

                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # ensure one second long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)

                    print(f"{file_path}: {i-1}")

    # store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
