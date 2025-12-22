import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from flamo.auxiliary.reverb import parallelFirstOrderShelving

def get_noise(noise_level_db: float, ir_len: int):
    e_target = np.sqrt(( 10 ** (noise_level_db / 10) ))
    noise_seq = np.random.randn(1, ir_len, 1)
    noise_seq = noise_seq * (1/np.sqrt(np.mean(np.pow(np.abs(noise_seq), 2))))
    return noise_seq * e_target

def get_target_t60(rt_DC, rt_Ny, fc, nfft, fs):

    delays = torch.tensor([1, 997, 1153, 1327, 1559, 1801, 2099])
    filter = parallelFirstOrderShelving(
        nfft=nfft,
        delays=delays,
        device="cpu",
        fs=fs,
        rt_nyquist=rt_Ny,
    )

    filter.assign_value(torch.tensor([rt_DC, fc]))

    x = torch.zeros((1, nfft, len(delays)), dtype=torch.float32)
    x[:, 0, :] = 1.0
    H = filter(torch.fft.rfft(x, dim=1, n=nfft))
    frequencies = torch.fft.rfftfreq(nfft, 1/fs) 
    indx = 1
    rt = - ( 60 * delays[indx] ) / ( fs * (20*torch.log10(torch.abs(H[0, :, indx])))) 
    f_bands =  [31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0]
    # find the values in rt that are closest to the f_bands
    f_bands = np.array(f_bands)
    mins_f = np.abs(f_bands - frequencies[:, None].numpy())
    index = np.argmin(mins_f, axis=0)
    rt_point = rt[index]
    print(rt_point)
    
    rt_all = - 60 / fs / (20*torch.log10(torch.abs(H[0, :, :]))) * delays
    error = torch.mean(torch.abs(rt_all[:, 0].unsqueeze(-1) - rt_all[:, 1:]))
    plt.plot(frequencies, rt_all[:, 0].numpy(), '--')
    plt.plot(frequencies, rt_all[:, 1:].numpy())
    plt.axvline(x=fc/2/torch.pi*fs)
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('RT60 [s]')
    plt.savefig('matlab/rt60_firstorder.png')

    # save the rt_points and frequencies in a mat file
    savemat('matlab/target_rt60.mat', {'frequencies': frequencies.numpy(), 'rt_point': rt_point.numpy()})

if __name__ == "__main__":
    nfft = 96000
    fs = 48000
    x = get_target_t60(2, 0.5, 2 * np.pi * 10000/48000, nfft=96000, fs=48000)
