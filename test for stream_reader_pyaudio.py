import time
import numpy as np
import matplotlib.pyplot as plt

from src.stream_reader_pyaudio import Stream_Reader


def main():
    # Create stream reader object with 2 channels
    stream_reader = Stream_Reader(channels=2)

    # Start the stream
    stream_reader.start_stream()

    # Initialize a variable to keep track of the number of iterations
    num_iterations = 0

    # Continuously plot the FFT
    while True:
        if stream_reader.new_data:
            # Compute the FFT of the last window of data
            fft = np.fft.rfft(stream_reader.data_buffer.data[-1], axis=0)

            # Plot the FFT
            plt.clf()
            plt.plot(np.abs(fft[:, 0]), label="Left Channel")
            plt.plot(np.abs(fft[:, 1]), label="Right Channel")
            plt.legend()
            plt.pause(0.001)

            # Increment the number of iterations
            num_iterations += 1

            # Exit the loop after a certain number of iterations
            if num_iterations == 100:
                break

    # Stop the stream
    stream_reader.stop_stream()


if __name__ == '__main__':
    main()
