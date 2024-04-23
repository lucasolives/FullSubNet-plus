from audiomentations import AddGaussianSNR, RoomSimulator, Mp3Compression, Compose
import numpy as np

class NoiseAugmentation:
    def __init__(self, sr) -> None:
        self.sr = sr

        self.min_gaussian_snr = 20
        self.max_gaussian_snr = 30
        self.gaussian_prob = 0.5

        self.min_x_room_size = 3.0
        self.max_x_room_size = 15.0
        self.min_y_room_size = 3.0
        self.max_y_room_size = 15.0
        self.min_z_room_size = 2.5
        self.max_z_room_size = 15.0
        self.min_x_position = 0.1
        self.max_x_position = 2.9
        self.min_y_position = 0.1
        self.max_y_position = 2.9
        self.min_z_position = 0.8
        self.max_z_position = 2.0
        self.min_mic_distance = 1.0
        self.max_mic_distance = 2.0
        self.room_prob = 0.5

        self.mp3_min_bitrate = 8
        self.mp3_max_bitrate = 64
        self.mp3_prob = 0.5

        self._augment_func = Compose([
            AddGaussianSNR(min_snr_db=self.min_gaussian_snr, max_snr_db=self.max_gaussian_snr, p=self.gaussian_prob),
            RoomSimulator(min_size_x=self.min_x_room_size, max_size_x=self.max_x_room_size,
                          min_size_y=self.min_y_room_size, max_size_y=self.max_y_room_size,
                          min_size_z=self.min_z_room_size, max_size_z=self.max_z_room_size,
                          min_source_x=self.min_x_position, max_source_x=self.max_x_position,
                          min_source_y=self.min_y_position, max_source_y=self.max_y_position,
                          min_source_z=self.min_z_position, max_source_z=self.max_z_position,
                          min_mic_distance=self.min_mic_distance, max_mic_distance=self.min_mic_distance,
                          leave_length_unchanged=True, p=self.room_prob),
            Mp3Compression(min_bitrate=self.mp3_min_bitrate, max_bitrate=self.mp3_max_bitrate, p=self.mp3_prob)
        ])

    def augment(self, audio : np.ndarray, sample_rate) -> np.ndarray:
        return self._augment_func(samples=audio, sample_rate=sample_rate)
