import numpy as np
import time, math, scipy
from collections import deque
from scipy.signal import savgol_filter

from src.fft import getFFT
from src.utils import *

from src.visualizer import Spectrum_Visualizer

class LeftEar:
    def __init__(self):
        self.n_frequency_bins = 50
        # Add other attributes and methods here as needed

class RightEar:
    def __init__(self):
        self.n_frequency_bins = 50
        # Add other attributes and methods here as needed

class Stream_Analyzer:
    """
    The Audio_Analyzer class provides access to continuously recorded
    (and mathematically processed) audio data.

    Arguments:

        device: int or None:      Select which audio stream to read .
        rate: float or None:      Sample rate to use. Defaults to something supported.
        FFT_window_size_ms: int:  Time window size (in ms) to use for the FFT transform
        updatesPerSecond: int:    How often to record new data.

    """

    def __init__(self,
        device = None,
        rate   = None,
        FFT_window_size_ms  = 50,
        updates_per_second  = 100,
        smoothing_length_ms = 50,
        n_frequency_bins    = 51,
        visualize = True,
        verbose   = False,
        height    = 450,
        window_ratio = 24/9):

        self.left_ear = LeftEar()
        self.right_ear = RightEar()

        self.n_frequency_bins = n_frequency_bins
        self.rate = rate
        self.verbose = verbose
        self.visualize = visualize
        self.height = height
        self.window_ratio = window_ratio

        try:
            from src.stream_reader_pyaudio import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = updates_per_second,
                verbose = verbose)
        except:
            from src.stream_reader_sounddevice import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = updates_per_second,
                verbose = verbose)

        self.rate = self.stream_reader.rate

        #Custom settings:
        self.rolling_stats_window_s    = 20     # The axis range of the FFT features will adapt dynamically using a window of N seconds
        self.equalizer_strength        = 0.20   # [0-1] --> gradually rescales all FFT features to have the same mean
        self.apply_frequency_smoothing = True   # Apply a postprocessing smoothing filter over the FFT outputs

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03*self.n_frequency_bins) - 1

        self.FFT_window_size = round_up_to_even(self.rate * FFT_window_size_ms / 1000)
        self.FFT_window_size_ms = 1000 * self.FFT_window_size / self.rate
        self.fft  = np.ones(int(self.FFT_window_size/2), dtype=float)
        self.fftx = np.arange(int(self.FFT_window_size/2), dtype=float) * self.rate / self.FFT_window_size

        self.data_windows_to_buffer = math.ceil(self.FFT_window_size / self.stream_reader.update_window_n_frames)
        self.data_windows_to_buffer = max(1,self.data_windows_to_buffer)

        # Temporal smoothing:
        # Currently the buffer acts on the FFT_features (which are computed only occasionally eg 30 fps)
        # This is bad since the smoothing depends on how often the .get_audio_features() method is called...
        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.FFT_window_size_ms, self.smoothing_length_ms, verbose=1)
            self.feature_buffer_left = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype=np.float32, data_dimensions=2)
            self.feature_buffer_right = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype=np.float32, data_dimensions=2)

        # This can probably be done more elegantly...
        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2, dtype=None) - 1
        self.fftx_bin_indices = np.round(((self.fftx_bin_indices - np.max(self.fftx_bin_indices))*-1) / (len(self.fftx) / self.n_frequency_bins), 0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)), self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies_left = np.zeros(self.n_frequency_bins)
        self.frequency_bin_energies_right = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres  = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin   = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        # Hardcoded parameters:
        self.fft_fps = 30
        self.log_features = False   # Plot log(FFT features) instead of FFT features --> usually pretty bad
        self.delays = deque(maxlen=20)
        self.num_ffts = 0
        self.strongest_frequency = 0

        # Assume the incoming sound follows a pink noise spectrum:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate/2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps # Assumes ~30 FFT features per second
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value = 25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        print("Using FFT_window_size length of %d for FFT ---> window_size = %dms" % (self.FFT_window_size, self.FFT_window_size_ms))
        print("##################################################################################################")

        # Let's get started:
        self.stream_reader.stream_start(self.data_windows_to_buffer)

        if self.visualize:
            self.visualizer = Spectrum_Visualizer(
                stream_analyzer=self,  # Pass 'self' as the Stream_Analyzer object
                ear_left=self.left_ear,
                ear_right=self.right_ear,
                height=self.height,
                frequency_bin_energies_left=self.frequency_bin_energies_left,
                frequency_bin_energies_right=self.frequency_bin_energies_right,
                window_ratio=self.window_ratio,
            )
            self.visualizer.start()

    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies_left)
        self.rolling_bin_values.append_data(self.frequency_bin_energies_right)
        self.bin_mean_values  = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values  = np.maximum((1 - self.equalizer_strength) * np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self, n_bins=3):

        latest_data_window = self.stream_reader.data_buffer.get_most_recent(self.FFT_window_size)

        latest_data_window_left = latest_data_window[::2]
        latest_data_window_right = latest_data_window[1::2]

        self.fft_left = getFFT(latest_data_window_left, self.rate, self.FFT_window_size, log_scale=self.log_features)
        self.fft_right = getFFT(latest_data_window_right, self.rate, self.FFT_window_size, log_scale=self.log_features)

        # Equalize pink noise spectrum falloff:
        self.fft_left = self.fft_left * self.power_normalization_coefficients
        self.fft_right = self.fft_right * self.power_normalization_coefficients

        self.num_ffts += 1
        self.fft_fps  = self.num_ffts / (time.time() - self.stream_reader.stream_start_time)

        if self.smoothing_length_ms > 0:
            self.feature_buffer_left.append_data(self.fft_left)
            self.feature_buffer_right.append_data(self.fft_right)

            buffered_features_left = self.feature_buffer_left.get_most_recent(len(self.smoothing_kernel))
            buffered_features_right = self.feature_buffer_right.get_most_recent(len(self.smoothing_kernel))

            if len(buffered_features_left) == len(self.smoothing_kernel) and len(buffered_features_right) == len(self.smoothing_kernel):
                buffered_features_left = self.smoothing_kernel * buffered_features_left
                buffered_features_right = self.smoothing_kernel * buffered_features_right

                self.fft_left = np.mean(buffered_features_left, axis=0)
                self.fft_right = np.mean(buffered_features_right, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft_left + self.fft_right)]

        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies_left[bin_index] = np.mean(self.fft_left[self.fftx_indices_per_bin[bin_index]])
            self.frequency_bin_energies_right[bin_index] = np.mean(self.fft_right[self.fftx_indices_per_bin[bin_index]])

        return

    def get_audio_features(self):

        if self.stream_reader.new_data:  # Check if the stream_reader has new audio data we need to process
            if self.verbose:
                start = time.time()

            self.update_features()
            self.update_rolling_stats()
            self.stream_reader.new_data = False

            self.frequency_bin_energies_left = np.nan_to_num(self.frequency_bin_energies_left, copy=True)
            self.frequency_bin_energies_right = np.nan_to_num(self.frequency_bin_energies_right, copy=True)

            if self.apply_frequency_smoothing:
                if self.filter_width > 3:
                    self.frequency_bin_energies_left = savgol_filter(self.frequency_bin_energies_left, self.filter_width, 3)
                    self.frequency_bin_energies_right = savgol_filter(self.frequency_bin_energies_right, self.filter_width, 3)

            self.frequency_bin_energies_left[self.frequency_bin_energies_left < 0] = 0
            self.frequency_bin_energies_right[self.frequency_bin_energies_right < 0] = 0

            if self.verbose:
                self.delays.append(time.time() - start)
                avg_fft_delay = 1000. * np.mean(np.array(self.delays))
                print("  FFT delay: {:.2f} ms".format(avg_fft_delay))
                print("  FFT fps: {:.1f}".format(self.fft_fps))

        return self.frequency_bin_energies_left, self.frequency_bin_energies_right

    def close(self):
        self.stream_reader.close()

def getFFT(data, rate, window_size, log_scale=True):
    fft_raw = np.fft.rfft(data * np.hanning(len(data)))
    fft_mag = np.abs(fft_raw)
    if log_scale:
        fft_mag = 20 * np.log10(fft_mag)
    return fft_mag

if __name__ == "__main__":
    import time

    device_index = None
    stream_reader = AudioStreamReader(device_index=device_index, rate=44100, chunk_size=1024)
    stream_reader.start()

    feature_extractor = AudioFeatureExtractor(stream_reader=stream_reader, smoothing_length_ms=20)

    while True:
        try:
            feature_extractor.get_audio_features()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break

    feature_extractor.close()
