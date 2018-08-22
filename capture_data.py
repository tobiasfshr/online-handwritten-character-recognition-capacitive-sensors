import time
from ops import get_current_input
import numpy as np
import matplotlib.pyplot as plt
import re
import ctypes
import pickle
import os.path


class DataObserver:
    def __init__(self, datapath):
        self.datapath = datapath
        data, raw_data = self.get_data()
        self.curr_len = len(data)
        self.curr_raw_len = len(raw_data)
        self.cache = []
        self.timesteps = 0
        self.timesteps_thresh = 5  # max. 250ms
        self.waiting = False

    def get_data(self):
        file = open(self.datapath, "r")
        raw_data = re.split(r'[\r\n]+', file.read().strip())
        file.close()
        data = get_current_input(raw_data)
        return data, raw_data

    def step(self):
        #get current data
        data, raw_data = self.get_data()

        #is the system waiting for more input?
        if self.waiting:
            # was there a new input?
            if len(data) != self.curr_len:
                self.cache += data[-1]
                self.timesteps = 0
                self.waiting = True
                self.curr_raw_len = len(raw_data)
                self.curr_len = len(data)
            # is the user currently inputting?
            if len(raw_data) != self.curr_raw_len:
                self.timesteps = 0
                self.waiting = True
                self.curr_raw_len = len(raw_data)
            # if timestep_thresh is reached, return cache
            if self.timesteps >= self.timesteps_thresh:
                self.waiting = False
                return self.cache
        else:
            # was there a new input?
            if len(data) != self.curr_len:
                #initialize cache and waiting params
                self.cache = data[-1]
                self.timesteps = 0
                self.waiting = True
                self.curr_raw_len = len(raw_data)
                self.curr_len = len(data)

        self.timesteps += 1
        return None


if __name__ == '__main__':
    # prediction for sample
    letter = input("Enter captured letter: ")
    savepath = './data/' + letter + '.pkl'
    observer = DataObserver("demo.log")
    dataset = []
    print("Capturing data.")
    while True:
        new_entry = observer.step()
        if new_entry != None:
            # add sequence to dataset
            signal = np.asarray(new_entry)
            plt.plot(signal[:, 1], signal[:, 0])
            plt.show()
            #messagedialog - store, discard, stop recording
            decision = ctypes.windll.user32.MessageBoxW(0, "data OK?", "data review", 6)  # 11 continue, 10 try again, 2 cancel
            if decision == 11:
                dataset.append(signal)
                print("number entries captured: ", len(dataset))
            if decision == 2:
                dataset.append(signal)
                if os.path.exists(savepath):
                    total_set = pickle.load(open(savepath, 'rb'))
                    dataset = total_set + dataset
                pickle.dump(dataset, open(savepath, 'wb'))
                print("number entries total: ", len(dataset))
                exit(0)

        time.sleep(0.05)
