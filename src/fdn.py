import numpy as np
import sympy as sp
import scipy
from collections import OrderedDict

import torch

from flamo.processor import dsp, system
from flamo.auxiliary.reverb import inverse_map_gamma, parallelFirstOrderShelving

def rt2slope(rt60: torch.Tensor, fs: int):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope.
    """
    return -60 / (rt60 * fs)


def rt2absorption(rt60: torch.Tensor, fs: int, delays_len: torch.Tensor):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope relative to the delay line length.
    """
    slope = rt2slope(rt60, fs)
    return torch.einsum("i,j->ij", slope, delays_len)

class CustomShell(system.Shell):
    def __init__(self, core, input_layer, output_layer):
        super().__init__(core, input_layer, output_layer)

    def forward(self, x: torch.Tensor, ext_param: dict = None) -> torch.Tensor:
        r"""
        Forward pass through the input layer, the core, and the output layer. Keeps the three components separated.

            **Args**:
                - x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                - torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        get_delays = self._Shell__core.feedback_loop.feedforward.delays.get_delays()
        delays = get_delays(self._Shell__core.feedback_loop.feedforward.delays.param)
        self._Shell__core.feedback_loop.feedforward.attenuation.delays = delays
        x = self._Shell__input_layer(x)
        if ext_param is not None:
            x = self._Shell__core(x, ext_param)
        else:
            x = self._Shell__core(x)
        x = self._Shell__output_layer(x)
        return x

class FDN:
    r"""
    Class for creating a Feedback Delay Network (FDN).
    The FDN is initialized with homogeneous attenuation.
    """

    def __init__(self, config_dict: dict, requires_grad: bool = True, filter_requires_grad: bool = True):

        # get config parameters
        self.config_dict = config_dict
        # get number of delay lines
        self.N = self.config_dict.N
        # get delay line lengths
        self.delays = self.config_dict.delays
        if self.delays is None:
            self.delays = self.get_delay_lines()
        # get device and dtype
        self.device = self.config_dict.device
        self.dtype = torch.float64
        # set requires_grad
        self.requires_grad = requires_grad
        self.filter_requires_grad = filter_requires_grad
        # create a random isntance of the FDN
        self.fdn = self.get_fdn_instance()
        self.set_model()

    def set_model(self, input_layer=None, output_layer=None):
        # set the input and output layers of the FDN model
        if input_layer is None:
            input_layer = dsp.FFT(self.config_dict.nfft, dtype=torch.float64)
        if output_layer is None:
            output_layer = dsp.iFFT(nfft=self.config_dict.nfft, dtype=torch.float64)

        self.model = self.get_shell(input_layer, output_layer)

    def get_fdn_instance(self):

        # delay lines
        delay_lines = torch.tensor(self.delays, device=self.config_dict.device)
        input_vector = np.empty((self.N,))
        input_vector[::2] = 1
        input_vector[1::2] = -1

        # Input and output gains
        input_gain = dsp.Gain(
            size=(self.N, 1),
            nfft=self.config_dict.nfft,
            device=self.config_dict.device,
            alias_decay_db=self.config_dict.alias_decay_db,
            dtype=torch.float64,
            requires_grad=self.requires_grad
        )
        input_gain.assign_value(torch.tensor(input_vector, device=self.config_dict.device).unsqueeze(1))
        output_gain = dsp.Gain(
            size=(1, self.N),
            nfft=self.config_dict.nfft,
            device=self.config_dict.device,
            alias_decay_db=self.config_dict.alias_decay_db,
            dtype=torch.float64,
            requires_grad=self.requires_grad
        )
        output_gain.assign_value(
            torch.ones((1, self.N), device=self.config_dict.device)
        )
        # RECURSION
        # Feedback loop with delays
        delays = dsp.parallelDelay(
            size=(self.N,),
            max_len=delay_lines.max(),
            nfft=self.config_dict.nfft,
            isint=True,
            device=self.config_dict.device,
            alias_decay_db=self.config_dict.alias_decay_db,
            requires_grad=False,
            dtype=torch.float64
        )
        # assign the required delay line lengths
        delays.assign_value(delays.sample2s(delay_lines))

        # feedback mixing matrix
        mixing_matrix = dsp.Matrix(
            size=(self.N, self.N),
            nfft=self.config_dict.nfft,
            matrix_type="orthogonal",
            device=self.config_dict.device,
            alias_decay_db=self.config_dict.alias_decay_db,
            dtype=torch.float64,
            requires_grad=self.requires_grad
        )

        # attenuation
        attenuation = self.get_attenuation(delay_lines)

        feedforward = system.Series(
            OrderedDict({"delays": delays, "attenuation": attenuation})
        )
        # Build recursion
        feedback_loop = system.Recursion(fF=feedforward, fB=mixing_matrix)

        # Build the FDN
        FDN = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )
        return FDN

    def get_homogeneous_attenuation(self, delay_lines):
        attenuation = dsp.parallelGain(
            size=(self.N,),
            nfft=self.config_dict.nfft,
            device=self.config_dict.device,
            alias_decay_db=self.config_dict.alias_decay_db,
            dtype=torch.float64, 
            requires_grad=self.filter_requires_grad
        )
        attenuation.map = map_gamma(delay_lines)
        if self.config_dict.rt60:
            inverse_gamma = inverse_map_gamma()
            gamma = inverse_gamma(
                torch.tensor(
                    10
                    ** (-3 / self.config_dict.sample_rate / self.config_dict.rt60[0]),
                    device=self.config_dict.device,
                )
            )
        else:
            gamma = 6
        attenuation.assign_value(
            gamma
            * torch.ones(
                (self.N,),
                device=self.config_dict.device,
            )
        )
        return attenuation

    def get_one_pole_attenuation(self, delay_lines):
        attenuation = parallelFirstOrderShelving(
            nfft=self.config_dict.nfft,
            delays=delay_lines,
            device=self.config_dict.device,
            fs=self.config_dict.sample_rate,
            rt_nyquist=self.config_dict.rt60,
            alias_decay_db=self.config_dict.alias_decay_db,
            dtype=torch.float64,
            requires_grad=self.filter_requires_grad
        )
        # overwrite the init_param method to set the initial parameters
        def init_param(self_inner):
            with torch.no_grad():
                # RT at dc
                self_inner.param[0] = 2
                self_inner.param[1] = 2 * torch.pi * 10000 / self_inner.fs  # shelf freq at 10kHz
        attenuation.init_param = init_param.__get__(attenuation)
        attenuation.init_param()
        return attenuation

    def get_attenuation(self, delay_lines):
        att_type = self.config_dict.attenuation_type
        if att_type == "homogeneous":
            return self.get_homogeneous_attenuation(delay_lines)
        elif att_type == "one_pole":
            return self.get_one_pole_attenuation(delay_lines)
        else:
            raise ValueError(f"Attenuation type {att_type} not supported")

    def get_shell(self, input_layer, output_layer):
        return CustomShell(
            core=self.fdn, input_layer=input_layer, output_layer=output_layer
        )
        
    def get_delay_lines(self):
        """Co-prime delay line lenghts for a given range"""
        ms_to_samps = lambda ms, fs: np.round(ms * fs / 1000).astype(int)
        delay_range_samps = ms_to_samps(
            np.asarray(self.config_dict.delay_range_ms), self.config_dict.sample_rate
        )
        # generate prime numbers in specified range
        prime_nums = np.array(
            list(sp.primerange(delay_range_samps[0], delay_range_samps[1])),
            dtype=np.int32,
        )
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        delay_lengths = np.array(
            np.r_[rand_primes[: self.N - 1], sp.nextprime(delay_range_samps[1])],
            dtype=np.int32,
        ).tolist()
        delay_lengths.sort()
        return delay_lengths

    def get_raw_parameters(self):
        # get the raw parameters of the FDN
        with torch.no_grad():
            core = self.model.get_core()
            param = {}
            param["A"] = core.feedback_loop.feedback.param.cpu().numpy()
            param["attenuation"] = (
                core.feedback_loop.feedforward.attenuation.param.cpu().numpy()
            )
            param["B"] = core.input_gain.param.cpu().numpy()
            param["C"] = core.output_gain.param.cpu().numpy()
            param["m"] = core.feedback_loop.feedforward.delays.param.cpu().numpy()
            return param

    def set_raw_parameters(self, param: dict):
        # set the raw parameters of the FDN from a dictionary
        with torch.no_grad():
            core = self.model.get_core()
            for key, value in param.items():
                try:
                    tensor_value = torch.tensor(value, device=self.config_dict.device)
                except:
                    continue
                if key == "A":
                    core.feedback_loop.feedback.assign_value(tensor_value)
                elif key == "B":
                    core.input_gain.assign_value(tensor_value.T)
                elif key == "C":
                    core.output_gain.assign_value(tensor_value)
                elif key == "m":
                    assert torch.equal(core.feedback_loop.feedforward.delays.param.squeeze(), tensor_value.squeeze()), "wrong delay line lengths"

            self.model.set_core(core)

    def read_parameters(self, filename):
        # read the parameters from a mat file
       return scipy.io.loadmat(filename)


    def normalize_energy(
        self,
        target_energy=1,
        is_time=False,
    ):
        """energy normalization done in the frequency domain
        Note that the energy computed from the frequency response is not the same as the energy of the impulse response
        Read more at https://pytorch.org/docs/stable/generated/torch.fft.rfft.html
        """

        H = self.model.get_freq_response(identity=False)
        if is_time:
            norm = (H.shape[1]-1)*2
            target_energy = target_energy * norm
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))
        target_energy = torch.tensor(target_energy, device=self.config_dict.device)
        # apply energy normalization on input and output gains only
        with torch.no_grad():
            core = self.model.get_core()
            core.input_gain.assign_value(
                torch.div(
                    core.input_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            core.output_gain.assign_value(
                torch.div(
                    core.output_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            self.model.set_core(core)

        # recompute the energy of the FDN
        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))
        assert (
            abs(energy_H - target_energy) / target_energy < 0.001
        ), "Energy normalization failed"

    def rt2gain(self, rt60):
        # convert RT60 to gain
        gdB = rt2absorption(
            rt60,
            self.config_dict.sample_rate,
            torch.tensor(self.delays, device=self.config_dict.device),
        ).squeeze()
        return 10 ** (gdB / 20)

    def get_max_dense_ir(self):
        cur_param = self.get_raw_parameters()
        dense_param = cur_param.copy()
        dense_param["B"] = np.ones((self.N, 1))
        dense_param["C"] = np.ones((1, self.N))
        dense_param["A"] = np.ones((self.N, self.N))
        core = self.model.get_core()
        for key, value in dense_param.items():
            try:
                tensor_value = torch.tensor(value, device=self.config_dict.device)
            except:
                continue
            if key == "A":
                core.feedback_loop.feedback.assign_value(tensor_value)
            elif key == "B":
                core.input_gain.assign_value(tensor_value)
            elif key == "C":
                core.output_gain.assign_value(tensor_value)
            elif key == "m":
                assert torch.equal(core.feedback_loop.feedforward.delays.param.squeeze(), tensor_value.squeeze()), "wrong delay line lengths"

        self.model.set_core(core)
        
        ir = self.model.get_time_response().cpu().numpy().squeeze()

        with torch.no_grad():
            core = self.model.get_core()
            for key, value in cur_param.items():
                try:
                    tensor_value = torch.tensor(value, device=self.config_dict.device)
                except:
                    continue
                if key == "A":
                    core.feedback_loop.feedback.assign_value(tensor_value)
                elif key == "B":
                    core.input_gain.assign_value(tensor_value)
                elif key == "C":
                    core.output_gain.assign_value(tensor_value)
                elif key == "m":
                    assert torch.equal(core.feedback_loop.feedforward.delays.param.squeeze(), tensor_value.squeeze()), "wrong delay line lengths"

            self.model.set_core(core)

        return ir

class map_gamma(torch.nn.Module):

    def __init__(self, delays):
        super().__init__()
        self.delays = delays.double()
        self.g_min = torch.tensor(0.99, dtype=torch.double, device=delays.device)
        self.g_max = torch.tensor(1, dtype=torch.double, device=delays.device)

    def forward(self, x):
        return (
            ((1 / (1 + torch.exp(-x[0]))) * (self.g_max - self.g_min) + self.g_min)
            ** self.delays
        ).type_as(x)