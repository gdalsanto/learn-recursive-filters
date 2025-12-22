import os
import time
from typing import (
    Dict, 
    Optional, 
    List, 
    Tuple
)

import torch
import numpy as np
import sympy as sp
from pydantic import (
    BaseModel, 
    model_validator, 
    ConfigDict, 
    field_validator
)

from flamo.optimize.surface import LossConfig, ParameterConfig

class LossProfileConfig(BaseModel):
    loss_config: LossConfig
    

class FDNConfig(BaseModel):
    """
    Configuration class.
    """
    # number of delay lines
    N: int = 6
    # sampling rate 
    sample_rate: int = 48000
    # number of fft points
    nfft: int = 96000
    # device to run the model
    device: str = 'cuda'
    # dtype 
    dtype: str = 'float64'
    # delays in samples
    delays: Optional[List[int]] = None
    # delay lengths range in ms
    delay_range_ms: List[float] = [15.0, 45.0]    
    # type of attenuation filter
    attenuation_type: str = 'homogeneous' # or 'geq'
    # reverberation time in seconds
    rt60: Optional[List[float]] = None
    # type of tone correction filter
    tone_type: str = None
    # tone filter linear gain level
    tone_gain: float = 1.0
    # colorless optimization
    is_colorless: bool = False

    alias_decay_db: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        if self.delays is None:
            self.delay_length_samps()

    def delay_length_samps(self) -> List[int]:
        """Co-prime delay line lenghts for a given range"""
        ms_to_samps = lambda ms, fs: np.round(ms * fs / 1000).astype(int)
        delay_range_samps = ms_to_samps(np.asarray(self.delay_range_ms),
                                        self.sample_rate)
        # generate prime numbers in specified range
        prime_nums = np.array(list(
            sp.primerange(delay_range_samps[0], delay_range_samps[1])),
                              dtype=np.int32)
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        self.delays = np.array(np.r_[rand_primes[:self.N - 1],
                                       sp.nextprime(delay_range_samps[1])],
                                 dtype=np.int32).tolist()
        self.delays.sort()
        return self.delays
        
    # check that N coincides with the length of delays or generate delays if 
    # they don't exists already
    @field_validator('delays', mode='after')
    @classmethod
    def check_delays_length(cls, v, values):
        if v is not None:
            if len(v) != values.data['N']:
                raise ValueError(f"Length of delays ({len(v)}) must match N ({values['N']})")
        return v
    
    # validator for training on GPU
    @field_validator('device', mode='after')
    @classmethod
    def validate_training_device(cls, value):
        """Validate GPU, if it is used for training"""
        if value == 'cuda':
            assert torch.cuda.is_available(
            ), "CUDA is not available for training"
    
    # forbid extra fields - adding this to help prevent errors in config file creation
    model_config = ConfigDict(extra="forbid")
    
class TrainerConfig(BaseModel):
    # training and validation split
    train_valid_split: float = 0.8
    # dataset size 
    num: int = 2**8
    # batch size for training
    batch_size: int = 1
    # maximum number of epochs
    max_epochs: int = 20
    # learning rate
    lr: float = 1e-3
    # minimum loss imporvenent to continue training
    patience_delta: float = 1e-2
    # step size of the scheduler
    step_size: int = 10
    # directory to save training results
    train_dir: Optional[str] = None
    # # loss function
    # loss: List[str] = ['mse'] NOTE not in use in this code
    # alpha for the loss function
    sparsity_weight: float = 1.0
    # whether or not learning the frequency independent parameters
    learn_independent_params: bool = False
    # check that the number of losses is equal to the number of alphas
    @model_validator(mode='before')
    @classmethod
    def check_loss_alpha_length(cls, values: Dict):
        """Check that the number of losses is equal to the number of alphas"""
        losses = values.get('loss')
        alphas = values.get('alpha')
        if losses is not None and alphas is not None:
            if len(losses) != len(alphas):
                raise ValueError("The number of losses must be equal to the number of alphas")
        return values
    
    def make_train_dir(self):
        """Make sure to save the training files in the right directory"""
        if self.train_dir is not None:
            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)
            else: 
                print("Warning: Directory already exists. Files may be overwritten.")
        else:
            self.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.train_dir)

class RoomGeometryConfig(BaseModel):
    num_rooms: int
    room_dims: List[List]
    start_coordinates: List[List]
    source_pos: List
    aperture_coords: Optional[List[List[Tuple]]] = None
    
class AmpGenerationConfig(BaseModel):
    # whether to generate amplitudes randomly, or based on geometry
    generate_amplitude_based_on_geometry: bool = False
    # path to pickle file containing amplitude distribution, analysed from the ThreeRoomDataset
    amp_dist_filepath: Optional[str] = None

class Config(BaseModel):
    fdn_config: FDNConfig
    target_fdn_config: FDNConfig
    loss_config: LossConfig
    param_config: ParameterConfig | List[ParameterConfig]
    device: str = 'cuda'
    target_type: str = 'wgn'
    indx_profile: Optional[int] = None
    add_noise_to_target: bool = False
    snr_db: float = None
    losses: List[List] = [
        ['MSS', {'func_name': 'mss_loss', 'args': {'name': 'MSS', 'apply_mask': True, 'alpha': 0}}],
        ['MSSnorm', {'func_name': 'mss_loss', 'args': {'name': 'MSSnorm','apply_mask': True, 'alpha': 0, 'energy_norm': True}}],
        ['MelMSS', {'func_name': 'mel_mss_loss', 'args': {'name': 'MelMSS', 'apply_mask': True, 'alpha': 0}}],
        ['MelMSSnorm', {'func_name': 'mel_mss_loss', 'args': {'name': 'MelMSSnorm','apply_mask': True, 'alpha': 0, 'energy_norm': True}}],
        ['AveragePower', {'func_name': 'AveragePower', 'args': {'name': 'AveragePower'}}],
        ['AveragePowerNorm', {'func_name': 'AveragePower', 'args': {'name': 'AveragePowerNorm', 'energy_norm': True}}],
        ['EDR', {'func_name': 'edr_loss', 'args': {'name': 'EDRnorm', 'energy_norm': True}}],
        ['EDC', {'func_name': 'edc_loss', 'args': {'name': 'EDCnorm', 'energy_norm': True, 'is_broadband': False, 'clip': False}}],
        ['EDCclip', {'func_name': 'edc_loss', 'args': {'name': 'EDCclipnorm', 'name': 'EDCclip', 'energy_norm': True, 'is_broadband': False, 'clip': True}}],
    ] 

class ConfigOptimization(BaseModel):
    fdn_config: FDNConfig
    target_fdn_config: FDNConfig
    trainer_config: TrainerConfig
    device: str = 'cuda'
    loss_weights: Optional[List[float]] = None
    target_params: Optional[List[float]] = None
    num_targets: Optional[int] = 1
    add_noise_to_target: bool = False
    snr_db: float = None
    losses: List[List] = [
        ['MSS', {'func_name': 'mss_loss', 'args': {'name': 'MSS', 'apply_mask': True, 'alpha': 0}}],
        ['MSSnorm', {'func_name': 'mss_loss', 'args': {'name': 'MSSnorm','apply_mask': True, 'alpha': 0, 'energy_norm': True}}],
        ['MelMSS', {'func_name': 'mel_mss_loss', 'args': {'name': 'MelMSS', 'apply_mask': True, 'alpha': 0}}],
        ['MelMSSnorm', {'func_name': 'mel_mss_loss', 'args': {'name': 'MelMSSnorm','apply_mask': True, 'alpha': 0, 'energy_norm': True}}],
        ['AveragePower', {'func_name': 'AveragePower', 'args': {'name': 'AveragePower'}}],
        ['AveragePowerNorm', {'func_name': 'AveragePower', 'args': {'name': 'AveragePowerNorm', 'energy_norm': True}}],
        ['EDR', {'func_name': 'edr_loss', 'args': {'name': 'EDRnorm', 'energy_norm': True}}],
        ['EDC', {'func_name': 'edc_loss', 'args': {'name': 'EDCnorm', 'energy_norm': True, 'is_broadband': False, 'clip': False}}],
        ['EDCclip', {'func_name': 'edc_loss', 'args': {'name': 'EDCclipnorm', 'name': 'EDCclip', 'energy_norm': True, 'is_broadband': False, 'clip': True}}],
    ] 