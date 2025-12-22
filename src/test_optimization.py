import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

import flamo.processor.dsp as dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.functional import signal_gallery
from flamo.optimize import sparsity_loss

from utils import *
from fdn import FDN
from config import ConfigOptimization
from target import get_noise
import losses


def main(config_dict):
    """
    Analyze the loss surface/profile for an FDN model.

    This function investigates how the loss varies with different parameter values,
    specifically the attenuation parameters of the FDN.

    Args:
        config_dict: Configuration dictionary containing FDN, loss, and optimization parameters
    """

    # ========== Configuration Setup ==========
    # Parse and prepare the configuration
    config = parse_config(config_dict)
    config["criteria"] = config_dict.losses
    # set the right random seed for reproducibility
    torch.manual_seed(84)
    np.random.seed(84)
    config["fdn"].delays = None  # force random co-prime delays
    config["target_fdn"].delays = None  # force random co-prime delays
    # ========== Save Configuration ==========
    # Write the configuration to a YAML file for reproducibility
    output_file = os.path.join(config["output_dir"], "config.yml")
    with open(output_file, "w") as file:
        yaml.dump(config, file)
    config["criteria"] = unpack_functions(
        losses, config_dict.losses, config_dict.device
    )

    # ========== Input Signal Generation ==========
    # Create an impulse signal
    input_signal = signal_gallery(
        signal_type="impulse",
        batch_size=1,
        n=1,
        n_samples=config["fdn"].nfft,
    )
    input_signal = torch.tensor(
        input_signal, dtype=torch.float64, device=config_dict.device
    )

    root_dir = config["output_dir"]
    rt_values = np.random.uniform(1, 3.5, size=config_dict.num_targets)
    fc_values = (
        2
        * np.pi
        * np.random.uniform(6000, 12000, size=config_dict.num_targets)
        / config["fdn"].sample_rate
    )
    # --- Trainer config_dict ---
    for i_target in range(config_dict.num_targets):
        # draw random target rt and fc parameters
        cur_rt = rt_values[i_target]
        cur_fc = fc_values[i_target]
        # update config dict with the number of target
        config["output_dir"] = os.path.join(root_dir, f"target_{i_target}")
        # make the directory
        os.makedirs(config["output_dir"], exist_ok=True)
        # save the target values
        scipy.io.savemat(
            os.path.join(config["output_dir"], "target_values.mat"),
            {"rt": cur_rt, "fc": cur_fc},
        )
        # Create the target FDN model for reference
        target_fdn = get_model(config_dict, config["target_fdn"], seed=84 + i_target)
        # set the target filter parameters
        target_fdn.model.get_core().feedback_loop.feedforward.attenuation.assign_value(
            torch.tensor(
                [cur_rt, cur_fc], dtype=torch.float64, device=config_dict.device
            )
        )
        # get the target FDN IR
        target_fdn_ir = target_fdn.model.get_time_response().detach()
        # compute energy of the target FDN IR
        energy_target_fdn = torch.mean(torch.pow(torch.abs(target_fdn_ir), 2))

        if config["add_noise_to_target"]:
            noise_level_lin = energy_target_fdn / (10 ** (config_dict.snr_db / 10))
            noise = torch.tensor(
                get_noise(
                    noise_level_db=10 * torch.log10(noise_level_lin).item(),
                    ir_len=target_fdn_ir.shape[1],
                ),
                dtype=torch.float64,
                device=config_dict.device,
            )
            target_fdn_ir = target_fdn_ir + noise
            scipy.io.savemat(
                os.path.join(config["output_dir"], "noise_term.mat"),
                {"noise_term": noise.cpu().numpy()},
            )
            # update criteria
            for i_crit in range(len(config["criteria"])):
                if config["criteria"][i_crit].add_noise:
                    config["criteria"][i_crit].noise_file = os.path.join(
                        config["output_dir"], "noise_term.mat"
                    )

        # ========== Create Dataset ==========
        # Generate dataset with input-target pairs and data augmentation
        dataset = Dataset(
            input=input_signal,
            target=target_fdn_ir,
            expand=config[
                "trainer_config"
            ].num,  # this coincides with the number of iterations gradient descent per epoch
        )
        train_loader, valid_loader = load_dataset(dataset, batch_size=1)
        print(f"Dataset created with {len(dataset)} samples")
        # save the target RIR
        rir_file = os.path.join(config["output_dir"], "target_rir.wav")
        sf.write(
            rir_file,
            target_fdn_ir.squeeze().detach().cpu().numpy(),
            config["fdn"].sample_rate,
        )
        print(f"Target RIR saved to {rir_file}")
        # Set up the FLAMO trainer with optimization parameters
        for i_crit, criterion in enumerate(config["criteria"]):
            print(f"Using loss criterion: {criterion.name}")

            # ========== FDN Model Initialization ==========
            # Create a FDN model with one-pole filter and anti-aliasing
            fdn = get_model(config_dict, config["fdn"], seed=84 + i_target)
            fdn.model.get_core().feedback_loop.feedforward.attenuation.assign_value(
                torch.tensor(
                    [cur_rt, cur_fc], dtype=torch.float64, device=config_dict.device
                )
            )
            fdn.normalize_energy(target_energy=energy_target_fdn, is_time=True)
            init_rt = np.random.uniform(1, 3.5, 1)
            init_fc = (
                2
                * np.pi
                * np.random.uniform(6000, 12000, 1)
                / config["fdn"].sample_rate
            )
            fdn.model.get_core().feedback_loop.feedforward.attenuation.assign_value(
                torch.tensor(
                    [init_rt, init_fc], dtype=torch.float64, device=config_dict.device
                ).squeeze()
            )
            # create a folder for the current criterion
            criterion_dir = os.path.join(config["output_dir"], criterion.name)
            if not os.path.exists(criterion_dir):
                os.makedirs(criterion_dir)

            trainer = Trainer(
                fdn.model,
                max_epochs=config["trainer_config"].max_epochs,
                lr=config["trainer_config"].lr,
                step_size=config[
                    "trainer_config"
                ].step_size,  # Learning rate scheduler step size
                patience_delta=config[
                    "trainer_config"
                ].patience_delta,  # Early stopping threshold
                train_dir=criterion_dir,  # Directory for saving checkpoints
                device=config_dict.device,
            )

            trainer.register_criterion(
                sparsity_loss(),  # Sparsity loss function
                alpha=config[
                    "trainer_config"
                ].sparsity_weight,  # Weight for sparsity loss
                requires_model=True,  # This loss needs access to model parameters
            )
            # Register the main criterion
            trainer.register_criterion(
                criterion, alpha=config["loss_weights"][i_crit], requires_model=False
            )

            # save initial parameters
            init_params = fdn.get_raw_parameters()
            param_file = os.path.join(criterion_dir, "init_parameters.mat")
            scipy.io.savemat(param_file, {"init_parameters": init_params})
            print(f"Initial parameters saved to {param_file}")

            # --- Training ---
            print(f"Starting FDN optimization with criterion: {criterion.name}...")
            trainer.train(train_dataset=train_loader, valid_dataset=valid_loader)
            print("Training completed!")

            # save the parameters of the trained model
            optimized_params = fdn.get_raw_parameters()
            optim_fdn_ir = fdn.model.get_time_response().detach()
            energy_fdn = torch.mean(torch.pow(torch.abs(optim_fdn_ir), 2))
            param_file = os.path.join(criterion_dir, "optim_parameters.mat")
            scipy.io.savemat(
                param_file,
                {
                    "optim_parameters": optimized_params,
                    "energy_target_fdn": energy_target_fdn.detach().cpu().numpy(),
                    "energy_optimized_fdn": energy_fdn.detach().cpu().numpy(),
                },
            )
            scipy.io.savemat(
                param_file.replace("parameters", "loss"), {"loss": trainer.train_loss}
            )
            print(f"Optimized parameters and loss saved to {param_file}")

            # save the final RIR
            with torch.no_grad():
                ir = fdn.model.get_time_response()
                ir = ir.detach().cpu().numpy()
                rir_file = os.path.join(criterion_dir, "optimized_rir.wav")
                sf.write(rir_file, ir.squeeze(), config["fdn"].sample_rate)
                print(f"Optimized RIR saved to {rir_file}")

            # print the final loss values and the parameters of the attenuation filter
            print(
                f"Final validation loss for criterion {criterion.name}: {trainer.valid_loss[-1]:.6f}"
            )
            core = fdn.model.get_core()
            attenuation_parameters = core.feedback_loop.feedforward.attenuation.param
            print(
                f"Optimized attenuation parameters: {attenuation_parameters.detach().cpu().numpy()}"
            )

            # print error
            if config["target_params"] is not None:
                target_params = np.array(config["target_params"])
                error = attenuation_parameters.detach().cpu().numpy() - target_params
                print(f"Error with respect to target parameter {error}")

            del trainer, fdn
        del target_fdn_ir, target_fdn, dataset, train_loader, valid_loader


def get_model(config_dict, config, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        # start from a random seed to make shure you're not always getting the same initialization
        torch.manual_seed(torch.randint(0, 10000).item())
        np.random.seed(torch.randint(0, 10000).item())
    fdn = FDN(config, requires_grad=config_dict.trainer_config.learn_independent_params)
    fdn.set_model(
        output_layer=dsp.iFFTAntiAlias(
            nfft=config.nfft,
            alias_decay_db=config.alias_decay_db,
            device=config_dict.device,
            dtype=torch.float64,
        )
    )
    return fdn


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
            config_dict = ConfigOptimization(**yaml.safe_load(file))
    else:
        # Use default configuration if no file specified
        config_dict = ConfigOptimization()

    # ========== Device Configuration ==========
    # Check for CUDA availability and fallback to CPU if needed
    if config_dict.device == "cuda" and not torch.cuda.is_available():
        config_dict.device = "cpu"
    print(f"Using device: {config_dict.device}")

    # ========== Output Directory Setup ==========
    # Determine output directory from command-line args or config
    if args.output_dir is not None:
        # Command-line argument overrides config file
        config_dict.trainer_config.train_dir = args.output_dir

    if config_dict.trainer_config.train_dir is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists(config_dict.trainer_config.train_dir):
            os.makedirs(config_dict.trainer_config.train_dir)
    else:
        # Generate timestamped output directory if none specified
        config_dict.trainer_config.train_dir = os.path.join(
            "output", time.strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(config_dict.trainer_config.train_dir)

    # ========== Execute Analysis ==========
    main(config_dict)
