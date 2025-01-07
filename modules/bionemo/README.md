# BIONEMO

## Setup

1. Run `docker.sh` to create a docker container. Peek inside the script to change certain options (i.e. pulling the latest image).
2. Inside the container, you have to install [NGC](https://org.ngc.nvidia.com/setup/installers/cli) according to instructions for AMD64 Linux.
3. Create `outputs` directory inside `modules/bionemo/data`.
4. Create a symbolic link between outputs and volume: `ln -s ../../modules/bionemo/data/outputs bionemo_sampling` while being in `data/sampling` directory.
