import numpy as np
import torch
from torch.utils.data import Dataset

class BLEPacketDataset(Dataset):
    def __init__(self, params, min_max=None):
        """
        min_max: optional dict like
          {
            "Sensor 1": (min1, max1),
            "Sensor 2": (min2, max2),
            "Sensor 3": (min3, max3),
          }
        If None, will fall back to per-window dynamic min/max (last resort).
        """
        self.sensors = {}         # Dictionary to store sensor readings by sensor name.
        self.total_data = None    # Will hold the post-processed (windowed) data.
        self.params = params
        self.min_max = min_max or {}


    def set_min_max(self, min_max):
        """Allow updating min/max after construction."""
        self.min_max = min_max or {}


    def add_packet(self, packet):
        """
        Add one packet worth of data into the dataset.

        Accepts either:
        • a pre-parsed dict: {"Sensor 1": val, ...}  -> append directly (no re-parse)
        • raw bytes/bytearray/hex str                -> parse, then append
        """
        # 1) Fast-path: pre-parsed sensors
        if isinstance(packet, dict):
            for sensor, value in packet.items():
                if sensor not in self.sensors:
                    self.sensors[sensor] = []
                self.sensors[sensor].append(value)
            return

        # 2) Back-compat: raw BLE payload as bytes/bytearray/hex str
        if isinstance(packet, (bytes, bytearray)):
            packet_hex = packet.hex()
        elif isinstance(packet, str):
            packet_hex = packet
        else:
            raise ValueError("Unsupported packet type. Must be bytes, bytearray, str, or dict.")

        sensors = self.parse_ble_packet(
            packet_hex,
            selected_channels=self.params.selected_channels,
        )
        for sensor, value in sensors.items():
            if sensor not in self.sensors:
                self.sensors[sensor] = []
            self.sensors[sensor].append(value)


    @staticmethod
    def parse_ble_packet (packet_hex, selected_channels = (1, )) -> dict:
        # Skip BLE header and split into 16-bit words (little-endian swap)
        packet_meaningful = packet_hex[2:]
        if len (packet_meaningful) % 4 != 0:
            raise ValueError ("The data does not break evenly into 4-character chunks.")
        channels_raw = [packet_meaningful[i: i + 4] for i in range (0, len (packet_meaningful), 4)]

        sensors = {}
        for idx, channel in enumerate (selected_channels, start = 1):
            index = channel - 1
            if index >= len (channels_raw):
                raise ValueError (f"Channel {channel} is not available in the packet.")
            chunk = channels_raw[index]
            reversed_chunk = chunk[2:] + chunk[:2]
            value = int (reversed_chunk, 16)
            sensors[f"Sensor {idx}"] = value
        return sensors


    def min_max_normalization (self):
        """
        Apply min–max normalization to the data from each sensor using
        externally provided min/max if available. If not available,
        fall back to per-sensor dynamic min/max computed from the current window.
        """
        for sensor, values in self.sensors.items ():
            values_array = np.array (values, dtype=float)

            if sensor in self.min_max:
                min_val, max_val = self.min_max[sensor]
            else:
                # Fallback: compute on the fly if pre-collected min/max was not provided.
                min_val = float(np.min(values_array)) if len(values_array) else 0.0
                max_val = float(np.max(values_array)) if len(values_array) else 1.0

            if max_val > min_val:
                normalized = (values_array - min_val) / (max_val - min_val)
            else:
                # Degenerate case (flat-lined channel). Leave unchanged.
                normalized = values_array
            
            normalized = np.clip(normalized, 0.0, 1.0)

            self.sensors[sensor] = normalized


    def smoothing(self, sorted_chunk, num_to_ignore):
        if len(sorted_chunk) <= 2 * num_to_ignore:
            return np.mean(sorted_chunk)
        trimmed = sorted_chunk[num_to_ignore: -num_to_ignore]
        return np.mean(trimmed) if len(trimmed) > 0 else 0


    def smoothing_sensor_values(self, window_size, num_to_ignore):
        total_data = {"sensors": []}
        if not self.sensors:
            raise ValueError("No sensor data available.")

        num_sensors = len(self.sensors)                # was fixed to 3
        sensor_keys = [f"Sensor {i}" for i in range(1, num_sensors + 1)]

        window_sensors = []
        for key in sensor_keys:
            sensor_window = self.sensors[key]
            chunk_size = 10
            chunks = [sensor_window[j:j + chunk_size] for j in range(0, window_size, chunk_size)]
            smooth_values = [self.smoothing(np.sort(np.array(chunk)), num_to_ignore) for chunk in chunks]
            window_sensors.append(np.array(smooth_values))
        total_data["sensors"].append(window_sensors)
        self.total_data = total_data



    def parsing(self):
        """
        Process the collected BLE packets into a dataset.
        Applies normalization, segmentation (via a sliding time window), smoothing, and labeling.
        """
        self.min_max_normalization()
        self.smoothing_sensor_values(self.params.window_size, self.params.num_to_ignore)
    
    def __len__(self):
        return 0 if self.total_data is None else len(self.total_data["sensors"])
    
    def __getitem__(self, index):
        # self.total_data["sensors"][index] is likely a list of arrays, shape (5, subchunks)
        sensors_list = self.total_data["sensors"][index]
        
        # If it is a Python list of np arrays, first stack them into one np array.
        # For example, shape = (5, subchunks)
        import numpy as np
        sensors_np = np.stack(sensors_list, axis = 0)  # shape (5, subchunks)
        
        # Convert that NumPy array into a PyTorch float tensor
        sensors_tensor = torch.from_numpy(sensors_np).float()
        return {"sensors": sensors_tensor}
