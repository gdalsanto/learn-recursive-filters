from typing import List

import numpy as np
import torch
import pyfar as pf
import scipy 
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import fftconvolve
from nnAudio import features


class mss_loss(nn.Module):
    r"""
    Multi-Scale Spectral Loss in the linear scale.
    This loss function computes the difference between predicted and true audio signals
    in the linear spectrogram domain across multiple FFT sizes.
    It is possible to apply a mask based on Signal-to-Noise Ratio (SNR) to the loss computation (this is particularly useful for training models on noisy data).
    The mask is calculated from the target (true) signal and is applied to both target and prediction before loss computation.
    The noise will be calculated as the mean of the energy of the last 0.01s unless its value is being passed as an argument.

    The loss is computed as the p norm of the difference between the predicted and true spectrograms.
    The spectrogram is computed using nnAudio's STFT class.

    Using the :arg:`form` argument, the loss can be computed in different ways:
    - **None**: The loss is computed as the p norm of the difference between the predicted and true spectrograms.
    - **yamamoto**: The loss is computed as the Frobenius norm of the difference between the predicted and true spectrograms, divided by the Frobenius norm of the true spectrogram. The log term is computed as the L1 norm of the difference between the predicted and true log spectrograms, divided by the number of elements in the true log spectrogram.
    - **magenta**: The loss is computed as the L1 norm of the difference between the predicted and true spectrograms, divided by the number of elements in the true spectrogram. The log term is computed as the L1 norm of the difference between the predicted and true log spectrograms, divided by the number of elements in the true log spectrogram.

    Attributes:
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
        - **name** (str): A name for the loss function. Default is "MelMSS".
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **apply_mask** (bool): Whether to apply a mask based on SNR. Default is False.
        - **threshold** (float): The SNR threshold for masking. Default is 5.
        - **p** (str): The order of the norm to be used. Default is "fro" (Frobenius norm).
        - **log_term** (bool): Whether to include the log term in the loss computation. Default is False.
        - **alpha** (float): A scaling factor for the log term in the loss computation. Default is 1.0.
        - **form** (str): The form of the loss to be used. Default is None.
        - **noise_energy** (float): The energy of the noise to be used for masking. Default is None.

    References:
        - Yamamoto, R., Song, E., & Kim, J. M. (2020, May). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6199-6203). IEEE.
        - Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP: Differentiable digital signal processing. arXiv preprint arXiv:2001.04643.
    """
    def __init__(
        self,
        nfft: List[int] = [128, 256, 512, 1024, 2048, 4096],
        overlap: float = 0.75,
        sample_rate: int = 48000,
        energy_norm: bool = False,
        t_mix: float = 0.0,
        device="cpu",
        name: str = "MSS",
        apply_mask: bool = False,
        threshold: float = 5,
        alpha_lin: float = 1.0,
        alpha_log: float = 1.0,
        noise_energy = None,
        f_min: float = 31.5,
        f_max: float = 20000,
        add_noise: bool = False,
        noise_file: str = None,
        snr: float = None,
    ):
        super().__init__()
        self.nfft = nfft
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.energy_norm = energy_norm
        self.name = name
        self.t_mix = t_mix
        self.device = device
        self.apply_mask = apply_mask
        self.threshold = threshold
        self.alpha_lin = alpha_lin
        self.alpha_log = alpha_log
        self.noise_energy = noise_energy
        self.f_min = f_min
        self.f_max = f_max
        self.add_noise = add_noise
        self.noise_file = noise_file
        self._lin_stft_fns = []
        self.snr = snr
        for n in self.nfft:
            hop_length = int(n * (1 - self.overlap))
            stft_fn = features.stft.STFT(
                n_fft=n,
                hop_length=hop_length,
                window="hann",
                freq_scale="linear",
                sr=self.sample_rate,
                fmin=self.f_min,
                fmax=self.f_max,
                output_format="Magnitude",
                verbose=False,
            ).to(self.device)
            self._lin_stft_fns.append(stft_fn)

    def forward(self, y_pred, y_true):
        if self.add_noise: 
            self.noise_term = torch.tensor(scipy.io.loadmat(self.noise_file)['noise_term']).to(self.device).to(torch.float64)
        if self.snr is not None:
            energy_pred = torch.mean(y_pred**2).item()
            energy_noise = energy_pred / (10**(self.snr/10))
            self.noise_term = self.noise_term * torch.sqrt(torch.tensor(energy_noise) / torch.mean(self.noise_term**2))
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"
        if self.add_noise:
            noise_term = self.noise_term.to(y_pred.device).to(y_pred.dtype)
            y_pred = y_pred + noise_term 

        n_channels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]

        n_remove = int(self.t_mix * self.sample_rate)
        y_pred = y_pred[:, n_remove:, :]
        y_true = y_true[:, n_remove:, :]

        if self.energy_norm:
            y_pred = y_pred / torch.norm(y_pred, p=2)
            y_true = y_true / torch.norm(y_true, p=2)

        # reshape it to (num_audio, len_audio) as indicated by nnAudio - accepts only torch.float32
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
        y_true = torch.reshape(y_true, (-1, y_true.shape[1]))

        loss = 0  # initialize match loss
        for i, nfft in enumerate(self.nfft):
            # initialize stft function with new nfft
            hop_length = int(nfft * (1 - self.overlap))
            lin_stft = self._lin_stft_fns[i]
            lin_stft = lin_stft.to(self.device).to(y_pred.dtype)

            h, w = tuple(lin_stft(y_pred).shape[-2:])
            lin_stft_pred = lin_stft(y_pred)
            lin_stft_true = lin_stft(y_true)
            # lin_stft_pred = lin_stft_pred / torch.sqrt((torch.sum(lin_stft_pred**2, -1).unsqueeze(-1)))
            # lin_stft_true = lin_stft_true / torch.sqrt((torch.sum(lin_stft_true**2, -1).unsqueeze(-1)))

            Y_pred_lin = torch.reshape(lin_stft_pred, (batch_size, h, w, n_channels))
            Y_true_lin = torch.reshape(lin_stft_true, (batch_size, h, w, n_channels))

            Y_pred_log = torch.reshape(torch.log(lin_stft_pred), (batch_size, h, w, n_channels))
            Y_true_log = torch.reshape(torch.log(lin_stft_true), (batch_size, h, w, n_channels))

            mask = torch.ones_like(Y_true_lin)
            if self.apply_mask:
                # TODO clean this up a little
                if not self.noise_energy:
                    # compute the noise energy as the mean of the last 0.01s
                    self.noise_energy_pred = torch.mean(
                        torch.pow(
                            Y_pred_lin[
                                :, :, -int(0.1 * self.sample_rate / hop_length), :
                            ],
                            2,
                        )
                    )
                    self.noise_energy_true = torch.mean(
                        torch.pow(
                            Y_true_lin[
                                :, :, -int(0.1 * self.sample_rate / hop_length), :
                            ],
                            2,
                        )
                    )
                else:
                    self.noise_energy_pred = self.noise_energy
                    self.noise_energy_true = self.noise_energy
                SNR_true = 10 * torch.log10(
                    torch.max(Y_true_lin**2, self.noise_energy_true * 1.01)
                    - self.noise_energy_true
                ) - 10 * torch.log10(self.noise_energy_true)
                # mask[(SNR_pred < self.threshold)] = 0
                mask[(SNR_true < self.threshold)] = 0
                N = torch.sum(mask)
            else:
                N = torch.numel(Y_true_lin)
            clip_indx = torch.nonzero(
                Y_true_log == torch.tensor(-float("inf"), device=self.device),
                as_tuple=True,
            )
            Y_pred_log[clip_indx] = torch.finfo(Y_pred_log.dtype).eps
            Y_true_log[clip_indx] = torch.finfo(Y_true_log.dtype).eps
            # update match loss
            loss += self.alpha_lin * torch.norm((Y_true_lin - Y_pred_lin) * mask, p="fro") / torch.norm(
                Y_true_lin, p="fro"
            ) + self.alpha_log * torch.norm((Y_true_log - Y_pred_log) * mask, p=1) / torch.numel(
                Y_true_log
            )
        return loss
 
## -------------------- ENERGY DECAY RELIEF LOSSES
class edr_loss(nn.Module):
    r"""
    Energy Decay Relief (EDR) Loss.

    This loss function computes the frequency-dependent loss on the mel-scale energy decay relief (EDR).

    Attributes:
        - **nfft** (int): The FFT size for the STFT computation. Default is 1024.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.5.
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
        - **name** (str): A name for the loss function. Default is "EDR".

    References:
        - Mezza, A. I., Giampiccolo, R., & Bernardini, A. (2024). Modeling the frequency-dependent sound energy decay of acoustic environments with differentiable feedback delay networks. In Proceedings of the 27th International Conference on Digital Audio Effects (DAFx24) (pp. 238-245).
    """
    def __init__(
        self,
        nfft: int = 1024,
        overlap: float = 0.5,
        sample_rate: int = 48000,
        t_mix: int = 0, 
        scale: str = "log",
        energy_norm: bool = False,
        device: str = "cpu",
        name: str = "EDR",
        f_min: float = 31.5,
        f_max: float = 20000,
        add_noise: bool = False,
        noise_file: str = None,
        snr: float = None,
    ):
        super().__init__()
        self.nfft = nfft
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.energy_norm = energy_norm
        self.t_mix = t_mix
        self.scale = scale
        self.win_length = int(0.020 * self.sample_rate)
        self.name = name
        self.device = device
        self.f_min = f_min
        self.f_max = f_max
        self._mel_stft = self.mel_stft()
        self.add_noise = add_noise
        self.noise_file = noise_file
        self.snr = snr
    def discard_last_n_percent(self, x, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * x.shape[1]))
        out = x[:, 0:last_id, :]

        return out

    def schroeder_backward_int(self, x):
        # expected shape (batch_size, h, w, n_channels)
        # Backwards integral
        out = torch.flip(x, dims=[-2])
        out = torch.cumsum(out**2, dim=-2)
        out = torch.flip(out, dims=[-2])

        # Normalize to 1
        if self.energy_norm:
            norm_vals = torch.max(out, dim=-2, keepdim=True)[0]  # per channel
        else:
            norm_vals = torch.ones(out.shape, device=out.device)

        out = out / norm_vals
        if self.scale == "log":
            out = 10 * torch.log10(out + 1e-32)  # avoid log(0)

        return out, norm_vals

    def mel_stft(self):
        # compute the mel spectrogram
        mel_stft = features.mel.MelSpectrogram(
            n_fft=self.nfft,
            hop_length=int(self.win_length * (1 - self.overlap)),
            window="hann",
            win_length=self.win_length,
            sr=self.sample_rate,
            fmin=self.f_min,
            fmax=self.f_max,
            n_mels=64,
            verbose=False,
        ).to(self.device)

        return mel_stft

    def forward(self, y_pred, y_true):
        if self.add_noise: 
            self.noise_term = torch.tensor(scipy.io.loadmat(self.noise_file)['noise_term']).to(self.device).to(torch.float64)
        if self.snr is not None:
            energy_pred = torch.mean(y_pred**2).item()
            energy_noise = energy_pred / (10**(self.snr/10))
            self.noise_term = self.noise_term * torch.sqrt(torch.tensor(energy_noise) / torch.mean(self.noise_term**2))
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"
        self._mel_stft = self._mel_stft.to(y_pred.dtype)
        n_channels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]
        if self.add_noise:
            noise_term = self.noise_term.to(y_pred.device).to(y_pred.dtype)
            y_pred = y_pred + noise_term 

        # reshape it to (num_audio, len_audio) as indicated by nnAudio
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
        y_true = torch.reshape(y_true, (-1, y_true.shape[1]))

        # remove the first t_mix seconds 
        n_remove = int(self.t_mix * self.sample_rate)
        y_pred = y_pred[:, n_remove:]
        y_true = y_true[:, n_remove:]

        h, w = tuple(self._mel_stft(y_pred).shape[-2:])
        Y_pred = torch.reshape(self._mel_stft(y_pred), (batch_size, h, w, n_channels))
        Y_true = torch.reshape(self._mel_stft(y_true), (batch_size, h, w, n_channels))

        Y_pred_edr = self.schroeder_backward_int(Y_pred)[0]
        Y_true_edr = self.schroeder_backward_int(Y_true)[0]

        # in case you get bad targets 
        clip_indx = torch.nonzero(
            Y_true_edr == torch.tensor(-float("inf"), device=self.device),
            as_tuple=True,
        )
        Y_true_edr[clip_indx] = torch.finfo(Y_true_edr.dtype).eps
        Y_pred_edr[clip_indx] = torch.finfo(Y_pred_edr.dtype).eps
        
        loss = torch.norm(Y_true_edr - Y_pred_edr, p=2) / torch.norm(Y_true_edr, p=2)
        return loss

    def analyze(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        n_channels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]
        # reshape it to (num_audio, len_audio) as indicated by nnAudio
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
        y_true = torch.reshape(y_true, (-1, y_true.shape[1]))

        h, w = tuple(self.mel_stft(y_pred).shape[-2:])
        Y_pred = torch.reshape(self.mel_stft(y_pred), (batch_size, h, w, n_channels))
        Y_true = torch.reshape(self.mel_stft(y_true), (batch_size, h, w, n_channels))

        Y_pred_edr = 10 * torch.log10(self.schroeder_backward_int(Y_pred)[0])
        Y_true_edr = 10 * torch.log10(self.schroeder_backward_int(Y_true)[0])

        # in case you get bad targets 
        clip_indx = torch.nonzero(
            Y_true_edr == torch.tensor(-float("inf"), device=self.device),
            as_tuple=True,
        )
        Y_true_edr[clip_indx] = torch.finfo(Y_true_edr.dtype).eps
        Y_pred_edr[clip_indx] = torch.finfo(Y_pred_edr.dtype).eps
        
        loss = torch.norm(Y_true_edr - Y_pred_edr, p=1) / torch.norm(Y_true_edr, p=1)
        return loss, Y_pred_edr, Y_true_edr
    

## -------------------- ENERGY DECAY CURVE LOSSES
class edc_loss(nn.Module):
    r"""
    Energy Decay Curve (EDC) Loss.

    This loss function computes the loss on energy decay curves (EDCs).
    It evaluates the similarity between the predicted and target EDCs, either in broadband or subband.

    Attributes:
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **nfft** (int): The FFT size. Default is 96000.
        - **is_broadband** (bool): Whether to compute the loss in broadband or subband. Default is False.
        - **n_fractions** (int): The number of fractional octave bands for subband analysis. Default is 1.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **clip** (bool): Whether to clip the EDCs at -60 dB. Default is False.
        - **name** (str): A name for the loss function. Default is "EDC".
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
    """
    def __init__(
        self,
        sample_rate: int = 48000,
        is_broadband: bool = False,
        n_fractions: int = 1,
        t_mix: int = 0, 
        energy_norm: bool = False,
        scale: str = "log",
        clip: bool = False,
        clip_level: float = -60,
        name: str = "EDC",
        device: str = "cpu",
        f_min: float = 31.5,
        f_max: float = 20000,
        add_noise: bool = False,
        noise_file: str = None,
        snr: float = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.is_broadband = is_broadband
        self.n_fractions = n_fractions
        self.energy_norm = energy_norm
        self.scale = scale
        self.t_mix = t_mix
        self.clip = clip
        self.clip_level = clip_level
        self.name = name
        self.device = device
        self.discard_n = 0.5
        self.f_min = f_min
        self.f_max = f_max
        self.mse = nn.MSELoss(reduction="mean")
        self._filter_cache = {} # cache for filterbank weights
        self.add_noise = add_noise
        self.noise_file = noise_file
        self.snr = snr

    def filterbank(self, x):
        _, L, _ = x.shape
        # cache key by input length
        key = int(L)

        if key not in self._filter_cache:
            # create an impulse of length L and compute fractional octave filters (pyfar)
            # This is done once for this input length and cached.
            impulse = np.zeros(key, dtype=np.float32)
            impulse[0] = 1.0
            # pyfar returns (filter_len, n_bands)
            filt_np = pf.dsp.filter.fractional_octave_bands(
                pf.Signal(impulse, self.sample_rate),
                num_fractions=self.n_fractions,
                frequency_range=(self.f_min, self.f_max),
            ).time.T

            self._filter_cache[key] = filt_np.squeeze() 


        # get weight and move to same device/dtype as x (avoid device transfers per band)
        weight = torch.from_numpy(self._filter_cache[key]).to(device=x.device, dtype=x.dtype)  # (n_bands,1,filter_len)
        y = torch.zeros(*x.shape, weight.shape[1], device=self.device)

        for i_band in range(weight.shape[-1]):
            y[..., i_band] = fftconvolve(x.transpose(2,1),
                                    weight[:, i_band].unsqueeze(0).unsqueeze(-1).repeat(x.shape[0],1,x.shape[0]).transpose(2,1),
                                    mode='full').transpose(2,1)[:, :x.shape[1], :]
    
        return y  # shape (batch, time, channels, n_bands)
    
    def discard_last_n_percent(self, x, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * x.shape[1]))
        out = x[:, 0:last_id, :]

        return out

    def schroeder_backward_int(self, x):

        # Backwards integral
        out = torch.flip(x, dims=[1])
        out = torch.cumsum(out**2, dim=1)
        out = torch.flip(out, dims=[1])

        # Normalize to 1
        if self.energy_norm:
            norm_vals = torch.max(out, dim=1, keepdim=True)[0]  # per channel
        else:
            norm_vals = torch.ones_like(out)

        out = out / norm_vals

        return out, norm_vals

    def get_edc(self, x):
        # Remove filtering artefacts (last 5 permille)
        out = self.discard_last_n_percent(x, self.discard_n)
        # compute EDCs
        if self.is_broadband:
            out = self.schroeder_backward_int(out)[0]
        else:
            out = self.schroeder_backward_int(self.filterbank(out))[0]
        # get energy in dB
        if self.scale == "log":
            out = 10 * torch.log10(out + 1e-32)
        return out

    def forward(self, y_pred, y_true):
        if self.add_noise: 
            self.noise_term = torch.tensor(scipy.io.loadmat(self.noise_file)['noise_term']).to(self.device).to(torch.float64)
        if self.snr is not None:
            energy_pred = torch.mean(y_pred**2).item()
            energy_noise = energy_pred / (10**(self.snr/10))
            self.noise_term = self.noise_term * torch.sqrt(torch.tensor(energy_noise) / torch.mean(self.noise_term**2))
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"
        if self.add_noise:
            noise_term = self.noise_term.to(y_pred.device).to(y_pred.dtype)
            y_pred = y_pred + noise_term
        # remove the first t_mix seconds
        n_remove = int(self.t_mix * self.sample_rate)
        y_pred = y_pred[:, n_remove:, :]
        y_true = y_true[:, n_remove:, :]
        # y_pred = torch.nn.functional.pad(y_pred, (0,0,0,n_remove))
        # compute the edcs
        y_pred_edc = self.get_edc(y_pred)
        y_true_edc = self.get_edc(y_true)

        if self.clip:
            try:
                if self.scale == "log":
                    clip_indx = torch.nonzero(
                        y_true_edc < (torch.max(y_true_edc, dim=1, keepdim=True)[0] + self.clip_level),
                        as_tuple=True,
                    )
                    y_pred_edc[clip_indx] = -180
                    y_true_edc[clip_indx] = -180
                else:
                    clip_indx = torch.nonzero(
                        y_true_edc < (torch.max(y_true_edc, dim=1, keepdim=True)[0] * 10**(self.clip_level / 10)),
                        as_tuple=True,
                    )
                    y_pred_edc[clip_indx] = 0
                    y_true_edc[clip_indx] = 0
            except:
                pass
        else:
            clip_indx = torch.nonzero(
                y_true_edc == torch.tensor(-float("inf"), device=self.device),
                as_tuple=True,
            )
            y_true_edc[clip_indx] = torch.finfo(y_true_edc.dtype).eps
            clip_indx = torch.nonzero(
                y_pred_edc == torch.tensor(-float("inf"), device=self.device),
                as_tuple=True,
            )
            y_pred_edc[clip_indx] = torch.finfo(y_pred_edc.dtype).eps

        # compute normalized mean squared error on the EDCs
        loss = torch.norm(y_true_edc - y_pred_edc, p=2) / torch.norm(y_true_edc, p=2)
        return loss