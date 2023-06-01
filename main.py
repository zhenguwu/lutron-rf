import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import asyncio
import math
np.set_printoptions(precision=2)

from rtlsdr import RtlSdr 

fsps = 2097152
fc = 433.65e6

def read_samples(center_freq, num_samples):
    sdr = RtlSdr()
    sdr.sample_rate = fsps
    sdr.center_freq = fc
    sdr.gain = 42.1
    #sdr.bandwidth = 500000

    print("We are using ")
    print("Gain (0==auto)  : ", sdr.gain)
    print("Sample Rate     : ", sdr.sample_rate)
    print("Center frequency: ", sdr.center_freq)

    # Read samples
    print("Reading samples...")
    samples = sdr.read_samples(num_samples)
    sdr.close()

    return samples

def plot_samples(samples, samples2 = None, lines = False):

    if samples2 is None:
        plt.plot(samples)
    else:
        (fig), (ax1, ax2) = plt.subplots(2)
        ax1.plot(samples)
        ax2.plot(samples2)
        if lines:
            for i in range(5, 11):
                ax1.axhline(y=(i/10.0), color='k')
                x2.axhline(y=(i/10.0), color='k')
    plt.show()



def clean_samples_FSK(samples, dsf=5):
    real = []
    for sample in samples:
        real.append(sample.real)
    return decimate(real[2000:], dsf)

# Cut samples to the next spike that is above a threshold
def cut_FSK(samples, threshold=0.5):
    # Find first spike
    i = 0
    while i < len(samples) and samples[i] < threshold:
        i += 1
    print("Cutting to next spike at index " + str(i))
    return samples[i:]

def find_middle_spike(samples, threshold=0.1):
    samples = cut_FSK(samples)
    notFound = True
    while notFound:
        # Cut to where first short spike would be after the tall spike
        samples = samples[4000:]
        # Find the start of the spike
        for i in range(0, 2000):
            if samples[i] > threshold:
                notFound = False
                return samples[i:i+825]

# Separate code to handle the pico's different signal pattern
def find_middle_spike_pico(samples, threshold=0.1):
    print("finding middle spike")
    samples = cut_FSK(samples)
    notFound = True
    while notFound:
        # Cut to where spike would be in between the tall spikes
        samples = samples[15000:]
        # Find the start of the spike if it exists, would be within 10,000 samples
        for i in range(0, 10000):
            if samples[i] > threshold:
                notFound = False
                return samples[i:i+700]
        # Was no short spike here, so we need to cut to the next tall spike and try again
        samples = cut_FSK(samples)

def slidingFreq(samples, windowSize=30, stepSize=5):
    freqs = []
    for i in range(0, len(samples) - windowSize, stepSize):
        window = samples[i:i+windowSize]
        crossings = 0
        for j in range(windowSize - 1):
            if window[j] < 0 and window[j+1] > 0 or window[j] > 0 and window[j+1] < 0:
                crossings += 1

        freqs.append(crossings)
    return freqs

def mapFreqsToVal(freqs):
    vals = []
    for freq in freqs:
        if freq > 8:
            vals.append(2)
        elif freq > 4:
            vals.append(1)
        else:
            vals.append(0)
    return vals

def filterVals(vals):
    valsCleaned = []
    for i in range(0, len(vals) - 9, 10):
        valsCleaned.append(np.rint(np.average(vals[i:i+10])))
    return valsCleaned

def classifyVals(vals):
    on = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0]
    off = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    if vals == on:
        print("On button pressed")
    elif vals == off:
        print("Off button pressed")
    else:
        print("Unknown button pressed")

def main():
    # Read samples from the SDR
    samples = read_samples(fc, fsps * 2)
    samples2 = read_samples(fc, fsps * 2)

    # Clean samples to just the real part and decimate
    samples = clean_samples_FSK(samples)
    samples2 = clean_samples_FSK(samples2)
    
    # Cut samples to just the short spike in between the tall spikes
    # This is for controlling the light with the app / Homekit
    samplesCut = find_middle_spike(samples)
    samplesCut2 = find_middle_spike(samples2)

    # Or for the pico remote / physical light switch
    #samplesCut = find_middle_spike_pico(samples)
    #samplesCut2 = find_middle_spike_pico(samples2)

    # Use a sliding window to find the relative frequency of the signal
    # [10:] is to align the changes in frequency with the sliding window
    freqs = slidingFreq(samplesCut[10:])
    freqs2 = slidingFreq(samplesCut2[10:])

    # Map the frequencies to three distinct values
    vals = mapFreqsToVal(freqs)
    vals2 = mapFreqsToVal(freqs2)

    # Average and round groups of values to remove noise
    valsClean = filterVals(vals)
    valsClean2 = filterVals(vals2)

    # Classify the bits as on or off
    print(valsClean)
    classifyVals(valsClean)
    print(valsClean2)
    classifyVals(valsClean2)

    # Plot the entire process
    plot_samples(samples, samples2)
    plot_samples(samplesCut, samplesCut2)
    plot_samples(freqs, freqs2)
    plot_samples(vals, vals2)
    plot_samples(valsClean, valsClean2)

if __name__ == "__main__":
    main()






# UNUSED CODE
# I first thought lutron was OOK/ASK but it's actually FSK

def clean_samples_ASK(samples, downscale, dsf=1000):
    samples_sq = [math.sqrt(i.real*i.real+i.imag*i.imag) for i in samples[2000:]]
    if downscale:
        samples_dec = decimate(samples_sq, dsf)
        return samples_dec
    return samples_sq

def process_samples_ASK(samples):
    tallSpike = []
    shortSpike = []
    tall = True
    i = 0
    while i < len(samples):
        if samples[i] < 0.2:
            while i < len(samples) and samples[i] <= 0.2:
                i += 1
        else:
            avg = 0
            N = 0
            while i < len(samples) and samples[i] > 0.2:
                avg += samples[i]
                N += 1
                i += 1
            if tall:
                tallSpike.append(avg / N)
            else:
                shortSpike.append(avg / N)
            tall = not tall
    return tallSpike, shortSpike