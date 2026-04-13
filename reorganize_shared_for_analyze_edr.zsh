#!/usr/bin/env zsh
set -euo pipefail

# Reorganize shared/ so analyze_edr_differences.m can find files at:
# shared/<snr>/target_<rir>/<loss>/<snr>_<noise_type>_<rir>_<loss>_optimized_rir_analysis.mat
#
# Current source layout is expected to be:
# shared/<snr>/<noise_type>/target_<rir>/<loss>/optimized_rir_analysis.mat
#
# Usage:
#   zsh reorganize_shared_for_analyze_edr.zsh --dry-run
#   zsh reorganize_shared_for_analyze_edr.zsh --apply
#   zsh reorganize_shared_for_analyze_edr.zsh --apply --mode symlink
#   zsh reorganize_shared_for_analyze_edr.zsh --apply --mode move
#
# Modes:
#   copy    : copy files into expected layout (safe, default)
#   symlink : create symlinks in expected layout
#   move    : move files into expected layout (destructive)

ROOT_DIR="shared"
MODE="copy"
APPLY=0

usage() {
  cat <<'EOF'
Reorganize shared folder for analyze_edr_differences.m

Options:
  --root <path>      Root directory to reorganize (default: shared)
  --mode <mode>      copy | symlink | move (default: copy)
  --apply            Perform changes (without this, script is dry-run)
  --dry-run          Explicit dry-run
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --dry-run)
      APPLY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "copy" && "$MODE" != "symlink" && "$MODE" != "move" ]]; then
  echo "Invalid --mode '$MODE'. Use copy, symlink, or move." >&2
  exit 2
fi

SNR_LEVELS=(
  snr10_bomb
  snr10_pb132
  snr10_se203
  snr20_bomb
  snr20_pb132
  snr20_se203
  snr10_synth
  snr20_synth
)

LOSS_TYPES=(
  EDC_lin
  EDC_log
  MSS
)

NOISE_TYPES=(
  noise_agnostic
  noise_aware
)

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root directory not found: $ROOT_DIR" >&2
  exit 1
fi

total_planned=0
total_done=0
total_skipped_existing=0
total_missing_source=0

do_copy() {
  local src="$1"
  local dst="$2"

  case "$MODE" in
    copy)
      cp -f "$src" "$dst"
      ;;
    symlink)
      ln -sfn "$(realpath "$src")" "$dst"
      ;;
    move)
      mv -f "$src" "$dst"
      ;;
  esac
}

echo "Root: $ROOT_DIR"
echo "Mode: $MODE"
if [[ $APPLY -eq 1 ]]; then
  echo "Run : APPLY"
else
  echo "Run : DRY-RUN"
fi

echo

for snr in $SNR_LEVELS; do
  snr_dir="$ROOT_DIR/$snr"
  if [[ ! -d "$snr_dir" ]]; then
    echo "[WARN] Missing SNR folder: $snr_dir"
    continue
  fi

  echo "== $snr =="

  target_dirs=$(find "$snr_dir"/noise_* -mindepth 1 -maxdepth 1 -type d -name 'target_*' 2>/dev/null | sed 's#^.*/##' | sort -u)

  if [[ -z "$target_dirs" ]]; then
    echo "  [WARN] No target_* directories found under $snr_dir/noise_*"
    echo
    continue
  fi

  for target_dir in ${(f)target_dirs}; do
    rir_name="${target_dir#target_}"

    for loss in $LOSS_TYPES; do
      dest_loss_dir="$snr_dir/$target_dir/$loss"

      for noise in $NOISE_TYPES; do
        src_file="$snr_dir/$noise/$target_dir/$loss/optimized_rir_analysis.mat"
        dst_file="$dest_loss_dir/${snr}_${noise}_${rir_name}_${loss}_optimized_rir_analysis.mat"

        total_planned=$((total_planned + 1))

        if [[ ! -f "$src_file" ]]; then
          echo "  [MISS] $src_file"
          total_missing_source=$((total_missing_source + 1))
          continue
        fi

        if [[ -e "$dst_file" ]]; then
          echo "  [SKIP] exists: $dst_file"
          total_skipped_existing=$((total_skipped_existing + 1))
          continue
        fi

        if [[ $APPLY -eq 1 ]]; then
          mkdir -p "$dest_loss_dir"
          do_copy "$src_file" "$dst_file"
          echo "  [OK] $src_file -> $dst_file"
          total_done=$((total_done + 1))
        else
          echo "  [PLAN] $src_file -> $dst_file"
        fi
      done
    done
  done

  echo
done

echo "Summary:"
echo "  Planned entries       : $total_planned"
echo "  Created/Moved/Linked  : $total_done"
echo "  Existing destination  : $total_skipped_existing"
echo "  Missing source files  : $total_missing_source"

if [[ $APPLY -eq 0 ]]; then
  echo
  echo "Dry-run only. Re-run with --apply to perform changes."
fi
