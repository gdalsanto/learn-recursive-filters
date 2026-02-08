import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import yaml

import flamo.processor.dsp as dsp
from flamo.functional import signal_gallery
from flamo.optimize.surface import LossSurface, LossProfile
from flamo.utils import save_audio

from config import Config
from fdn import FDN
import losses
from target import get_noise
from utils import parse_config, unpack_functions


def main(config_dict):
    """
    Analyze the loss surface/profile for an FDN model.

    This function investigates how the loss varies with different parameter values,
    specifically the attenuation parameters of the FDN.

    Args:
        config_dict: Configuration dictionary containing FDN, loss, and optimization parameters
    """
    # ========== Configuration Setup ==========
    # Set the ranodm seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ========== Configuration Setup ==========
    # Parse and prepare the configuration
    config = parse_config(config_dict)
    config["loss"].criteria = config_dict.losses

    # ========== FDN Model Initialization ==========
    # Create a FDN model with one-pole filter and anti-aliasing
    fdn = FDN(config["fdn"])
    fdn.set_model(
        output_layer=dsp.iFFTAntiAlias(
            nfft=config["fdn"].nfft,
            alias_decay_db=config["fdn"].alias_decay_db,
            device=config_dict.device,
            dtype=torch.float64,
        )
    )
    # ========== Save Configuration ==========
    # Write the configuration to a YAML file for reproducibility
    output_file = os.path.join(config["loss"].output_dir, "config.yml")
    with open(output_file, "w") as file:
        yaml.dump(config, file)
    config["loss"].criteria = unpack_functions(
        losses, config_dict.losses, config_dict.device
    )

    # ========== Input Signal Generation ==========
    # Create an impulse signal so that we can the IR
    input_signal = signal_gallery(
        signal_type="impulse",
        batch_size=1,
        n=1,
        n_samples=config["fdn"].nfft,
    )
    input_signal = torch.tensor(
        input_signal, dtype=torch.float64, device=config_dict.device
    )

    # ========== Loss Surface/Profile Setup ==========
    # Create either a 1D profile or 2D surface depending on configuration
    if config["indx_profile"] is not None:
        # 1D loss profile: vary a single parameter
        config["loss"].param_config = [config["param"][config["indx_profile"]]]
        loss_surface = LossProfile(
            fdn.model, config["loss"], device=config_dict.device, dtype=torch.float64
        )
    else:
        # 2D loss surface: vary two parameters
        config["loss"].param_config = config["param"]
        loss_surface = LossSurface(
            fdn.model, config["loss"], device=config_dict.device, dtype=torch.float64
        )

    # Set the raw parameters for the two dimensions being analyzed
    loss_surface.set_raw_parameter(
        config["param"][0].key,
        torch.tensor(config["param"][0].target_value),
        config["param"][0].param_map,
        config["param"][0].indx,
    )
    loss_surface.set_raw_parameter(
        config["param"][1].key,
        torch.tensor(config["param"][1].target_value),
        config["param"][1].param_map,
        config["param"][1].indx,
    )

    # ========== Target Signal Generation ==========
    fdn_target_ir = fdn.model(input_signal).detach()
    energy_fdn = torch.mean(torch.pow(torch.abs(fdn_target_ir), 2))
    # Save the target FDN impulse response
    with torch.no_grad():
        save_audio(
            os.path.join(config["loss"].output_dir, "fdn_target_ir.wav"),
            fdn_target_ir.squeeze().float(),
            fs=config["fdn"].sample_rate,
        )
        # save also the raw values
        scipy.io.savemat(
            os.path.join(config["loss"].output_dir, "fdn_target_ir.mat"),
            {"fdn_target_ir": fdn_target_ir.cpu().numpy()},
        )
    print(f"Random FDN delays: {config['target_fdn'].delays}")
    fdn_rnd = FDN(config["target_fdn"])
    fdn_rnd.set_model(
        output_layer=dsp.iFFTAntiAlias(
            nfft=config["target_fdn"].nfft,
            alias_decay_db=config["target_fdn"].alias_decay_db,
            device=config_dict.device,
            dtype=torch.float64,
        )
    )

    loss_surface_rnd = LossProfile(
        fdn_rnd.model, config["loss"], device=config_dict.device, dtype=torch.float64
    )
    loss_surface_rnd.set_raw_parameter(
        config["param"][0].key,
        torch.tensor(config["param"][0].target_value),
        config["param"][0].param_map,
        config["param"][0].indx,
    )
    loss_surface_rnd.set_raw_parameter(
        config["param"][1].key,
        torch.tensor(config["param"][1].target_value),
        config["param"][1].param_map,
        config["param"][1].indx,
    )
    fdn_rnd.normalize_energy(energy_fdn.item(), is_time=True)
    target_signal = fdn_rnd.model(input_signal).detach()
    scipy.io.savemat(
        os.path.join(config["loss"].output_dir, "fdn_target_ir_gt_clean.mat"),
        {"fdn_target_ir": target_signal.cpu().numpy()},
    )
    if config["add_noise_to_target"]:
        noise_level_lin = energy_fdn / (10 ** (config_dict.snr_db / 10))
        noise = torch.tensor(
            get_noise(
                noise_level_db=10 * torch.log10(noise_level_lin).item(),
                ir_len=target_signal.shape[1],
            ),
            dtype=torch.float64,
            device=config_dict.device,
        )
        target_signal = target_signal + noise
    # Save the noise signal in mat in the main directory
    if config["add_noise_to_target"]:
        scipy.io.savemat(
            os.path.join(config["loss"].output_dir, "noise_term_ref.mat"),
            {"noise_term_ref": noise.cpu().numpy()},
        )
        # generate another instance for the learnable FDN
        noise = torch.tensor(
            get_noise(
                noise_level_db=10 * torch.log10(noise_level_lin).item(),
                ir_len=target_signal.shape[1],
            ),
            dtype=torch.float64,
            device=config_dict.device,
        )
        scipy.io.savemat(
            os.path.join(config["loss"].output_dir, "noise_term.mat"),
            {"noise_term": noise.cpu().numpy()},
        )
    # Save the target FDN impulse response
    save_audio(
        os.path.join(config["loss"].output_dir, "fdn_target_ir_gt.wav"),
        target_signal.squeeze().float(),
        fs=config["fdn"].sample_rate,
    )
    # save also the raw values
    scipy.io.savemat(
        os.path.join(config["loss"].output_dir, "fdn_target_ir_gt.mat"),
        {"fdn_target_ir": target_signal.cpu().numpy()},
    )

    # Compute losses between FDN target IR and WGN target
    loss_target_ir = {}
    for i_crit in range(len(loss_surface.criteria)):
        criterion_name = loss_surface.criteria[i_crit].name
        loss_target_ir[criterion_name] = (
            loss_surface.criteria[i_crit](fdn_target_ir, target_signal)
            .cpu()
            .detach()
            .numpy()
        )

    # ========== Loss Computation ==========
    output_file = os.path.join(config["loss"].output_dir, "loss.mat")
    loss = loss_surface.compute_loss(input_signal, target_signal)
    loss_dict = {}
    loss_dict["loss_name"] = []
    for i_crit in range(len(loss_surface.criteria)):
        loss_dict[loss_surface.criteria[i_crit].name] = (loss[..., i_crit],)
        loss_dict["loss_name"].append(str(loss_surface.criteria[i_crit].name))
    loss_dict["loss"] = loss
    scipy.io.savemat(output_file, loss_dict)

    # save the minimum RIR for each criterion
    for i_crit in range(len(loss_surface.criteria)):
        criterion_name = loss_surface.criteria[i_crit].name
        min_idx = np.argmin(loss[:, i_crit]).item()
        # set the parameters to the minimum values
        loss_surface.set_raw_parameter(
            config["param"][config["indx_profile"]].key,
            loss_surface.steps[min_idx],
            config["param"][config["indx_profile"]].param_map,
            config["param"][config["indx_profile"]].indx,
        )

        min_rir = loss_surface.net.get_time_response().detach()
        # save the minimum rir
        save_audio(
            os.path.join(config["loss"].output_dir, f"fdn_min_ir_{criterion_name}.wav"),
            min_rir.squeeze().float(),
            fs=config["fdn"].sample_rate,
        )
        # save also the raw values
        scipy.io.savemat(
            os.path.join(config["loss"].output_dir, f"fdn_min_ir_{criterion_name}.mat"),
            {"fdn_min_ir": min_rir.cpu().numpy()},
        )

    fig, ax = loss_surface.plot_loss(loss)


if __name__ == "__main__":
    """
    Main entry point for the script.

    Handles command-line arguments, loads configuration, and executes the analysis.
    """

    # ========== Argument Parsing ==========
    parser = argparse.ArgumentParser(
        description="Analyze loss surface/profile for FDN models"
    )

    parser.add_argument(
        "--output_dir", type=str, help="Directory to save loss plots and results"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yml",
        help="Path to YAML configuration file for the loss profile analysis",
    )

    args = parser.parse_args()

    # ========== Configuration Loading ==========
    if args.config_file:
        # Resolve the relative file path to absolute path
        file_path = Path(args.config_file).resolve()
        # Read and parse the YAML configuration file
        with open(file_path, "r") as file:
            config_dict = Config(**yaml.safe_load(file))
    else:
        # Use default configuration if no file specified
        config_dict = Config()

    # ========== Device Configuration ==========
    # Check for CUDA availability and fallback to CPU if needed
    if config_dict.device == "cuda" and not torch.cuda.is_available():
        config_dict.device = "cpu"
    print(f"Using device: {config_dict.device}")

    # ========== Output Directory Setup ==========
    # Determine output directory from command-line args or config
    if args.output_dir is not None:
        # Command-line argument overrides config file
        config_dict.loss_config.output_dir = args.output_dir

    if config_dict.loss_config.output_dir is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists(config_dict.loss_config.output_dir):
            os.makedirs(config_dict.loss_config.output_dir)
    else:
        # Generate timestamped output directory if none specified
        config_dict.loss_config.output_dir = os.path.join(
            "output", time.strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(config_dict.loss_config.output_dir)

    # ========== Execute Analysis ==========
    main(config_dict)
