import scipy
import numpy as np 


def unpack_functions(module, lst: list, device='cpu'):
    """
    Unpack a list of functions.
    """
    functions = []
    for i_fun in range(len(lst)):
        # Extract the function name and arguments
        func_name = lst[i_fun][1]['func_name']
        args = lst[i_fun][1]['args']

        # Assume the function is defined in the current scope (use eval or globals())
        func = getattr(module, func_name)  # Or use eval(func_name) if safe

        # Call the function with unpacked arguments
        functions.append(func(**args, device=device))
    
    return functions

def parse_config(config_dict: dict):
    device = config_dict.device
    fdn_config = config_dict.fdn_config
    fdn_config.device = device
    target_fdn_config = config_dict.target_fdn_config
    target_fdn_config.device = device

    try :
        loss_config = config_dict.loss_config
        param_config = config_dict.param_config
        # create a dictionary 
        config = {
            "fdn": fdn_config,
            "target_fdn": target_fdn_config,
            "param": param_config,
            "loss": loss_config,
            "indx_profile": config_dict.indx_profile,
            "add_noise_to_target": config_dict.add_noise_to_target,
            "snr_db": config_dict.snr_db,
        }
        return config
    except AttributeError:
        # create a dictionary 
        config = {
            "fdn": fdn_config,
            "target_fdn": target_fdn_config,
            "criteria": None,
            "trainer_config": config_dict.trainer_config,
            "output_dir": config_dict.trainer_config.train_dir,
            "loss_weights": config_dict.loss_weights,
            "target_params": config_dict.target_params,
            "add_noise_to_target": config_dict.add_noise_to_target,
            "snr_db": config_dict.snr_db,
        }
        return config
    
def compute_echo(ir: np.ndarray, fs: int, N: int = 1024, preDelay: int = 0, mixingThresh: float = 0.9):
    """
    Computes the mixing time and echo density of an impulse response (IR).

    Args:
        ir (numpy.ndarray): The impulse response signal.
        fs (int): The sampling frequency of the IR.
        N (int, optional): The analysis window length. Defaults to 1024.
        preDelay (int, optional): The pre-delay of the IR. Defaults to 0.

    Returns:
        tuple: A tuple containing the mixing time (in milliseconds) and the echo density.

    Raises:
        ValueError: If the length of the IR is shorter than the analysis window length.

    References:
    Threshold of normalized echo density at which to determine "mixing time"
    Abel & Huang (2006) uses a value of 1.
    Pytorch translation of echoDensity.m from https://github.com/SebastianJiroSchlecht/fdnToolbox
    """
    # preallocate
    s = np.zeros(len(ir))
    echo_dens = np.zeros(len(ir))

    wTau = np.hanning(N)
    wTau = wTau / np.sum(wTau)

    halfWin = N // 2

    if len(ir) < N:
        raise ValueError('IR shorter than analysis window length (1024 samples). Provide at least an IR of some 100 msec.')

    sparseInd = np.arange(0, len(ir), 500)
    for n in sparseInd:
        # window at the beginning (increasing window length)
        # n = 1 to 513
        if n <= halfWin + 1:
            hTau = ir[0:n + halfWin]
            wT = wTau[-halfWin - n:]

        # window in the middle (constant window length)
        # n = 514 to end-511
        elif n > halfWin + 1 and n <= len(ir) - halfWin + 1:
            hTau = ir[n - halfWin:n + halfWin]
            wT = wTau

        # window at the end (decreasing window length)
        # n = (end-511) to end
        elif n > len(ir) - halfWin + 1:
            hTau = ir[n - halfWin:]
            wT = wTau[:len(hTau)]

        else:
            raise ValueError('Invalid n Condition')

        # standard deviation
        s[n] = np.sqrt(np.sum(wT * (hTau ** 2)))

        # number of tips outside the standard deviation
        tipCt = np.abs(hTau) > s[n]

        # echo density
        echo_dens[n] = np.sum(wT * tipCt)

    # normalize echo density
    echo_dens = echo_dens / scipy.special.erfc(1 / np.sqrt(2))

    echo_dens = np.interp(np.arange(1, len(ir) + 1), sparseInd, echo_dens[sparseInd])

    # determine mixing time
    d = np.argmax(echo_dens > mixingThresh)
    t_abel = (d - preDelay) / fs * 1000

    if t_abel is None:
        t_abel = 0
        print('Mixing time not found within given limits.')

    return t_abel, echo_dens