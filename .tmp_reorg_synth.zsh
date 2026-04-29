#!/usr/bin/env zsh
set -euo pipefail

root="/Users/dalsag1/Aalto Dropbox/Gloria Dal Santo/aalto/projects/learn-recursive-filters/website/audio"

for snr in snr10_synth snr20_synth; do
  src_root="$root/$snr"
  [[ -d "$src_root" ]] || continue

  for noise in noise_agnostic noise_aware; do
    src_noise="$src_root/$noise"
    [[ -d "$src_noise" ]] || continue

    for target_dir in "$src_noise"/target_*; do
      [[ -d "$target_dir" ]] || continue

      target_name="${target_dir:t}"
      rir_name="${target_name#target_}"
      dest_target="$src_root/$target_name"
      mkdir -p "$dest_target"

      if [[ -f "$target_dir/target_rir.wav" ]]; then
        cp -f "$target_dir/target_rir.wav" "$dest_target/target_rir.wav"
      fi
      if [[ -f "$target_dir/target_rir_analysis.mat" ]]; then
        cp -f "$target_dir/target_rir_analysis.mat" "$dest_target/target_rir_analysis.mat"
      fi
      if [[ -f "$target_dir/noise_term.mat" ]]; then
        cp -f "$target_dir/noise_term.mat" "$dest_target/noise_term.mat"
      fi

      for loss in MSS EDC_lin EDC_log; do
        src_loss="$target_dir/$loss"
        [[ -d "$src_loss" ]] || continue

        dest_loss="$dest_target/$loss"
        mkdir -p "$dest_loss"

        prefix="${snr}_${noise}_${rir_name}_${loss}"

        for kind in optimized_rir optimized_noisy_rir; do
          if [[ -f "$src_loss/$kind.wav" ]]; then
            cp -f "$src_loss/$kind.wav" "$dest_loss/${prefix}_${kind}.wav"
          fi
        done

        for kind in optimized_rir_analysis optimized_noisy_rir_analysis; do
          if [[ -f "$src_loss/$kind.mat" ]]; then
            cp -f "$src_loss/$kind.mat" "$dest_loss/${prefix}_${kind}.mat"
          fi
        done
      done
    done
  done
done
