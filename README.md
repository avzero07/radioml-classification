# radioml-classification
Modulation Classification Using Neural Networks

# Objective
Use Neural Networks (ANN and CNN based approaches) to perform modulation classification on received symbols.

# Datasets
A synthetic dataset, generated with GNU Radio, consisting of 11 modulations (8 digital and 3 analog) at varying signal-to-noise ratios. The dataset has the size 220,000×2×128, which means that there are 220,000 entries, each consisting of an array of size 2 × 128. Each array represents the samples of about 128 µs of a received waveform sampled with approximately 106 samples/second, and it contains between 8 and 16 modulation symbols. Since the samples of the signal waveforms are complex-valued, they have been stored as real and imaginary parts, and therefore we have arrays of size 2 × 128 in the data set.

The labels of the dataset contain two parameters:  

(1) The modulation technique used (one of [’8PSK’, ’AM-DSB’, ’AM-SSB’, ’BPSK’,’CPFSK’, ’GFSK’, ’PAM4’, ’QAM16’, ’QAM64’, ’QPSK’, ’WBFM’], so 11 possible modulation techniques)  

(2) The signal-to-noise ratio (SNR) value (one of [−20, −18, −16, −14, −12, −10, −8, −6, −4, −2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18], so 20 possible SNR values). The SNR is a measure for the quality of the communication channel. The higher the SNR, the less “noisy” is the channel.  

**Source		:** https://www.deepsig.io/datasets  
**Standard Dataset 	:** RML2016.10a.tar.bz2  
**Extended Dataset	:** RML2016.10b.tar.bz2  

# Framework
**Tensorflow		:** v 2.1.0  
**Keras			:** v 2.2.4-tf  
