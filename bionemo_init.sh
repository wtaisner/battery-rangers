#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Install NGC CLI (if not already installed) ---
if ! command -v ngc &> /dev/null; then
    echo "NGC CLI not found. Installing..."
    # Download and install the AMD64 Linux version as per the docs
    wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.6.6/files/ngccli_linux.zip -O ngccli_linux.zip
    unzip ngccli_linux.zip
    chmod u+x ngc-cli/ngc
    echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile
    source ~/.bash_profile
    echo "NGC CLI installed successfully."
else
    echo "NGC CLI is already installed."
fi


# --- 2. Configure BioNeMo and Authenticate NGC CLI ---
echo "Configuring BioNeMo and NGC CLI..."
# Copy the host .env file to the correct location
cp /tmp/.env_host /workspace/bionemo/.env
# Load the variables from the .env file into the current shell
source /workspace/bionemo/.env

# Check for the API key before proceeding
if [ -z "${NGC_CLI_API_KEY}" ]; then
    echo "ERROR: NGC_CLI_API_KEY is not set in your .env file. Cannot configure NGC."
    exit 1
fi

# We are piping the required inputs (API key, format, org, team, ace)
# separated by newlines into the command.
printf "%s\n\n%s\n%s\n\n" \
    "${NGC_CLI_API_KEY}" \
    "${NGC_CLI_ORG}" \
    "${NGC_CLI_TEAM}" \
    | ngc config set

echo "NGC CLI configured successfully."

# --- 2.5. Install Python Dependencies ---
echo "--- Installing Python Dependencies ---"

# Install the 'docker' library from PyPI
if ! pip show docker &> /dev/null; then
    echo "Installing 'docker' Python library from PyPI..."
    pip install docker
else
    echo "'docker' Python library is already installed."
fi

# Install the local 'MolRL' package in editable mode
# The path inside the container is /workspace/bionemo/data/MolRL
MOLRL_PATH="/workspace/bionemo/data/MolRL"
if [ -d "${MOLRL_PATH}" ]; then
    echo "Found MolRL at ${MOLRL_PATH}. Installing in editable mode..."
    # We must 'cd' into the directory to run 'pip install -e .'
    (cd "${MOLRL_PATH}" && pip install -e .)
else
    echo "WARNING: MolRL directory not found at ${MOLRL_PATH}. Skipping editable install."
fi
echo "--- Python Dependencies Installed ---"

# --- 3. Download Specific Model (MolMIM) if it doesn't exist ---
BIONEMO_HOME="/workspace/bionemo"
MODEL_NAME="molmim_70m_24_3"
MODEL_DIR="${BIONEMO_HOME}/models"
CHECKPOINT_FILE="${MODEL_DIR}/${MODEL_NAME}.nemo"

if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "MolMIM checkpoint not found at ${CHECKPOINT_FILE}. Downloading..."
    mkdir -p "${MODEL_DIR}"
    cd "${BIONEMO_HOME}"

    # This command will now run with the correctly authenticated CLI
    python download_artifacts.py --model_dir "${MODEL_DIR}" --models "${MODEL_NAME}"

    echo "Model download complete."
else
    echo "MolMIM checkpoint already exists. Skipping download."
fi


# --- 4. Start Jupyter Lab ---
echo "Initialization complete. Starting Jupyter Lab..."
exec jupyter lab --allow-root --ip=* --port=8899 --no-browser \
  --NotebookApp.token='' --NotebookApp.allow_origin='*' \
  --ContentsManager.allow_hidden=True --notebook-dir=/workspace/bionemo
