# Use the official miniconda3 image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the container
COPY environment.yaml .

# Create the conda environment
RUN conda env create -f environment.yaml

# Activate the conda environment
SHELL ["conda", "run", "-n", "battery", "/bin/bash", "-c"]

# Copy the rest of the application code to the container
COPY . .
ENV PYTHONPATH=/app
# Run the tests
CMD ["conda", "run", "-n", "battery", "pytest"]
CMD ["conda", "run", "-n", "battery", "pylint", "modules"]
