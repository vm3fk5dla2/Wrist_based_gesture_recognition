import pandas as pd
import numpy as np 
from torch.utils.data import Dataset


class SensorDataset (Dataset):
    def __init__ (self, params):
        self.sensors = {"sensor1": [], "sensor2": []}
        self.total_index = []
        self.labels = []
        self.total_data = []
        self.params = params


    def load_file (self, file):
        try:
            data = pd.read_csv(file, usecols=["Counts", "GestreIdentifier", "Channel 2", "Channel 3"], encoding='ISO-8859-1')
            data = data.dropna(how="all")
        except UnicodeDecodeError:
            # If the above encoding does not work, you can try using 'latin1' or 'cp1252'
            data = pd.read_csv(file, usecols=["Counts", "GestreIdentifier", "Channel 2", "Channel 3"], encoding='ISO-8859-1')
            data = data.dropna(how="all")

        for cnt, lbl, s1, s2 in zip (
            data["Counts"], data["GestreIdentifier"],
            data["Channel 2"], data["Channel 3"]):
            if lbl == 30:
                continue
            self.total_index.append(cnt)
            self.labels.append(lbl)
            self.sensors["sensor1"].append(s1)      # thumb finger
            self.sensors["sensor2"].append(s2)      # index finger



    def min_max_normalization (self):
        for (sensor_name, sensor_value) in self.sensors.items ():
            sensor_value = np.array (sensor_value)
            min_sensor_value = np.amin (sensor_value)
            max_sensor_value = np.amax (sensor_value)
            self.sensors[sensor_name] = (sensor_value - min_sensor_value) / (max_sensor_value - min_sensor_value)



    def smoothing (self, chunked_list, num_to_ignore):
        trimmed_list = np.sort (chunked_list)[num_to_ignore: -num_to_ignore]
        
        return sum (trimmed_list) / len (trimmed_list) if len (trimmed_list) > 0 else 0


    def sliding_timewindow_and_smoothing (self, window_size, num_to_ignore):
        total_data = {"sensors": [], "label": []}
        total_index = {"counts": self.total_index, "gesture_inits": []}

        for i in range (0, len (self.sensors["sensor1"]) - window_size + 1, self.params.stride):
            if any(curr != prev + 1 for prev, curr in zip(self.total_index[i: i + window_size], self.total_index[i: i + window_size][1:])):
                continue
            
            total_index["gesture_inits"].append (self.total_index[i])
            
            empty_list = list ()
            for sensor_value in self.sensors.values ():
                sensor_list = sensor_value[i: i + window_size]
                chunked_list = [sensor_list[j: j + 10] for j in range (0, window_size, 10)]
                sorted_chunked_list = [np.sort (list (chunk)) for chunk in chunked_list]
                smooth_list = np.array ([self.smoothing (chunk, num_to_ignore) for chunk in sorted_chunked_list])
                empty_list.append (smooth_list)

            total_data["sensors"].append (empty_list)
            total_data["label"].append (self.labels[i: i + window_size])
        
        self.total_data = total_data
        self.total_index = total_index


    def labeling (self, label_threshold, pinch_threshold):
        new_labels = []
        only_gestures = []
        only_gestures_index = []
        i = 0
        for labels in (self.total_data["label"]):
            only_gestures.append (self.total_data["sensors"][i])
            only_gestures_index.append (self.total_index["gesture_inits"][i])
            # LEFT
            if labels.count (9) >= label_threshold:
                new_labels.append (1)
            
            # RIGHT
            elif labels.count (3) >= label_threshold:
                new_labels.append (2)
            
            # PINCH
            elif labels.count (20) >= pinch_threshold:
                new_labels.append (3)
            
            # REST
            elif labels.count (0) >= label_threshold:
                new_labels.append (0)
            
            else:
                only_gestures.pop ()
                only_gestures_index.pop ()
            i += 1
        
        self.total_data["label"] = new_labels
        self.total_data["sensors"] = only_gestures
        self.total_index["gesture_inits"] = only_gestures_index

    def parsing (self):
        self.min_max_normalization ()
        self.sliding_timewindow_and_smoothing (self.params.window_size, self.params.num_to_ignore)
        self.labeling (self.params.label_threshold, self.params.pinch_threshold)

    def __len__ (self):
        return len (self.total_data["label"])
    
    def __getitem__ (self, index):
        item = {"sensors": self.total_data["sensors"][index], "label": self.total_data["label"][index]}
        return item