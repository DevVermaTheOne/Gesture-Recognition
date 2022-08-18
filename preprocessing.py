import numpy as np
import librosa
import json
import os

json_path = "data.json"

data_dict = {
    "mappings": ["hand at rest", "hand clenched", "wrist flexion",
                 "wrist extension", "radial deviation", "ulnar deviation"],
    "label": [],
    "MFCCs": []
}

# iterating through files

n = 0  # counting number of files passed
for subdir, dirs, files in os.walk("dataset"):
    for file in files:
        n = n + 1
        if n < 71:  # leaving last two files for prediction
            print("file", n)
            file_path = subdir + os.sep + file
            # loading file to a nparray
            data = np.genfromtxt(file_path, delimiter='')

            # currently only using channel 1 data and labels
            data = data[:, [1, 9]]

            # stacking data based on labels
            stacked_data = data[data[:, 1] == 0]
            stacked_data = stacked_data[:3000, 0]
            if len(stacked_data) < 3000:  # making sure array has 3000 values
                for j in range(len(stacked_data), 3000):  # filling with -1 so that i can remove them later
                    np.append(stacked_data, -1)

            stacked_data = stacked_data[np.newaxis, ...]

            for i in range(1, 7):
                sep = data[data[:, 1] == i]
                sep = sep[:3000, 0]
                if len(sep) < 3000:  # making sure array has 3000 values
                    for j in range(len(sep), 3000):  # filling with -1 so that i can remove them later
                        sep = np.append(sep, -1)
                stacked_data = np.vstack((stacked_data, sep[np.newaxis, ...]))

            # calculating mfccs
            sr = 1000
            n_fft = 128
            hop_length = 32

            for i in range(1, 7):
                signal = stacked_data[i]
                if -1 in signal:  # rejecting arrays containing -1
                    pass
                else:
                    MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=6, n_mels=20)
                    data_dict["label"].append(i)
                    data_dict["MFCCs"].append(MFCCs.T.tolist())



with open(json_path, 'w') as fp:
    json.dump(data_dict, fp, indent=4)