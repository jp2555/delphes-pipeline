#!/bin/bash
#SBATCH --job-name=delphes-validate
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.log
#
# Run the Delphes card-validation gate on Perlmutter.
#   sbatch runners/validate.sbatch [path/to/config.yml]
#
# Exit code is the gate: 0 = PASS, non-zero = FAIL.
set -euo pipefail

CONFIG="${1:-delphes_pipeline/validation/config.example.yml}"

# Use the project's pixi env (Perlmutter's bare `python` is 2.7). Override with
# PYTHON=... if you manage the environment yourself (e.g. a module/venv).
PYTHON="${PYTHON:-pixi run python}"
$PYTHON -m delphes_pipeline.validation.run_validation --config "$CONFIG"
