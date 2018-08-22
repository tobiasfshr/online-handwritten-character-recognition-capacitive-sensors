import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from ops import add_features

# variables and dimensions
numSeqs = 0
numDims = 1
numTimesteps = 0
inputPattSize = 3  # x, y and pen-up

path = './unipen_data/train_r01_v07/data/2/'
data_path = './unipen_data/train_r01_v07/include/'
save_path = './data/'
vocabulary = 'PEAWSB'


def is_data_point(inputString):
    point_list = inputString.split(' ')
    if len(point_list) == 2:
        for point in point_list:
            if not all(char.isdigit() or char == ' ' for char in point):
                return False
        return True
    else:
        return False


def get_data_file(file_content):
    for line in file_content:
        line = line.replace('\n', '')
        if line.find('.INCLUDE') != -1 and line.find('.dat') != -1:
            return open(os.path.join(data_path, line.replace('.INCLUDE ', '')), "r")


# Process all active samples in the sets
X = []
y = []
pen_up = []
for writer in os.listdir(path):
    writer_path = os.path.join(path, writer)
    for root, dirs, files in os.walk(writer_path):
        for session in files:
            print("processing file: ", root + session)
            session_file_lines = open(os.path.join(root, session), "r").readlines()
            data_file = get_data_file(session_file_lines)
            strokes = []
            current_stroke = []
            for line in data_file.readlines():
                line = line.replace('\n', '')
                if line.find('.PEN_UP') != -1:
                    strokes.append(current_stroke)
                    current_stroke = []
                elif is_data_point(line):
                    stroke_point = line.split(' ')
                    current_stroke.append([float(stroke_point[0]), float(stroke_point[1]), 0.])

            for line in session_file_lines:
                if line.find('.SEGMENT CHARACTER') != -1:
                    if line.find(' ? ') != -1:
                        index_and_label = line.split('?')
                    else:
                        index_and_label = line.split('OK')

                    char_idx = index_and_label[0]
                    char_idx = char_idx.replace('.SEGMENT CHARACTER ', '').replace(' ', '')
                    char_idx = char_idx.split('-')
                    char_idx = list(map(int, char_idx))
                    if len(char_idx) > 1:
                        char_idx = list(range(char_idx[0], char_idx[-1]+1))

                    label = index_and_label[1].replace(' ', '').replace('"', '').replace('\n', '')

                    sequence = []
                    for idx in char_idx:
                        sequence.extend(strokes[idx])
                        sequence.extend([[0., 0., 1.]])

                    # add features
                    sequence = np.asarray(sequence)
                    pen_up.append(sequence[:, 2])
                    sequence = sequence[:, 0:2]
                    sequence = add_features(sequence)

                    onehot_label = np.zeros(len(vocabulary)+1)
                    if label.upper() in vocabulary:
                        X.append(sequence)
                        onehot_label[vocabulary.index(label.upper())] = 1.
                        y.append(onehot_label)
                    else:
                        X.append(sequence)
                        onehot_label[-1] = 1.
                        y.append(onehot_label)

# normalize data and add pen_up column
unipen_data_scaler = StandardScaler()
unipen_data_scaler.fit(np.vstack(X))
for i in range(len(X)):
    sequence = np.asarray(X[i])
    sequence = unipen_data_scaler.transform(sequence)
    X[i] = np.column_stack((sequence, pen_up[i])).tolist()

# Save to file
pickle.dump(X, open(save_path + 'unipen_X.pkl', 'wb'))
pickle.dump(y, open(save_path + 'unipen_y.pkl', 'wb'))
pickle.dump(unipen_data_scaler, open(save_path + 'unipen_scaler.pkl', 'wb'))
print("done.")



