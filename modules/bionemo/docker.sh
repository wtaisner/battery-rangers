# https://docs.nvidia.com/bionemo-framework/1.10/access-startup.html

# Use NGC CLI to pull the latest version of the bionemo image
# NGC install: https://org.ngc.nvidia.com/setup/installers/cli
# Can also check manually: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/tags (much easier)

#LATEST_TAG=$(ngc registry image list --format_type=csv --column=tag nvidia/clara/bionemo-framework | tail +2 | head -1 | cut -f 2 -d',')
LATEST_TAG=1.10

BIONEMO_IMAGE_PATH=nvcr.io/nvidia/clara/bionemo-framework:${LATEST_TAG}

#docker pull "$BIONEMO_IMAGE_PATH" # Skip if already pulled

# Runs the container with the following options:
# --gpus all: Use all GPUs available on the machine
# --name bionemo: Name of the container
# ipc, ulimit are recommended settings for running the container
#
docker run --gpus all --name bionemo --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d -p 8888:8888 \
  -v ./data:/workspace/bionemo/data "$BIONEMO_IMAGE_PATH" \
  "jupyter lab --allow-root --ip=* --port=8888 --no-browser \
  --NotebookApp.token='' --NotebookApp.allow_origin='*' \
  --ContentsManager.allow_hidden=True --notebook-dir=/workspace/bionemo"
