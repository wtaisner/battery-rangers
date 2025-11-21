"""
Provides a high-level interface to run Continuous Symmetry Measure (CSM) calculations.

See: https://github.com/continuous-symmetry-measure/csm?tab=readme-ov-file for details.

This module defines two main components:
1.  `MoleculeCSMAnalysis`: A dataclass to store the aggregated results of CSM
    calculations for a single molecule across multiple symmetry point groups.
2.  `CSMRunner`: A class that manages all interactions with a Docker container
    running the 'teamcsm/csm' tool. It handles container lifecycle,
    file I/O, conformer generation via an external utility, and parsing of
    results into the `MoleculeCSMAnalysis` structure.
"""

import os
import shutil
import uuid
from dataclasses import dataclass, field

import docker
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.utils import compute_conformer


@dataclass
class MoleculeCSMResult:
    """
    Holds the complete CSM analysis for a molecule across various point groups.

    This dataclass acts as a structured container for all data generated during
    the analysis of a single SMILES string.

    Attributes:
        smiles: The input SMILES string representing the molecule.
        csm_results: A dictionary mapping each point group (e.g., 'c2') to its
            calculated CSM value (float).
        input_sdf_path: The absolute path to the generated 3D SDF input file.
        output_dirs: A dictionary mapping each point group to the path of its
            corresponding output directory created by the CSM tool.
        error_messages: A dictionary capturing any errors that occurred during
            the calculation for a specific point group.
    """

    smiles: str
    csm_results: dict[str, float] = field(default_factory=dict)
    input_sdf_path: str | None = None
    output_dirs: dict[str, str] = field(default_factory=dict)
    error_messages: dict[str, str] = field(default_factory=dict)

    @property
    def lowest_csm_normalized(self) -> tuple[str, float] | None:
        """
        Normalizes the lowest CSM value by the number of atoms in the molecule.

        Returns:
            A tuple containing the point group name (str) and its corresponding
            normalized CSM value (float) for the best fit, or None if no successful
            results are available.
        """
        if not self.csm_results:
            return None

        # Normalize CSM values by the number of atoms in the molecule
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return None

        normalized_lowest_csm = self.lowest_csm[1] / mol.GetNumAtoms()
        return self.lowest_csm[0], normalized_lowest_csm

    @property
    def lowest_csm(self) -> tuple[str, float] | None:
        """
        Finds the point group with the minimum CSM value.

        This property provides a convenient way to identify the symmetry group
        that the molecule most closely matches, based on the principle that a
        lower CSM value indicates a better fit.

        Returns:
            A tuple containing the point group name (str) and its corresponding
            CSM value (float) for the best fit, or None if no successful

            results are available.
        """
        if not self.csm_results:
            return None
        return min(self.csm_results.items(), key=lambda item: item[1])


class CSMRunner:
    def __init__(
        self,
        image_name: str = "teamcsm/csm:latest",
        container_name: str = "csm_runner_default",
        local_dir: str = "./csm_workspace",
    ) -> None:
        self.client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name

        # 1. Local Workspace: Where Python writes files
        self.local_workspace = os.path.abspath(local_dir)

        # 2. Host Workspace: What path the Docker Daemon on the host should mount.
        # If you are inside a container/DevContainer, set 'CSM_HOST_MOUNT_PATH'
        # to the actual path on your HOST machine.
        self.host_workspace = os.getenv("CSM_HOST_MOUNT_PATH", self.local_workspace)
        self.container_internal_data_dir = "/data"

        if not os.path.exists(self.local_workspace):
            os.makedirs(self.local_workspace, exist_ok=True)

    def _get_or_start_container(self, pull_image: bool) -> docker.models.containers.Container:
        if pull_image:
            try:
                self.client.images.get(self.image_name)
            except ImageNotFound:
                self.client.images.pull(self.image_name)

        try:
            container = self.client.containers.get(self.container_name)

            # Check for stale mounts (if the directory path changed)
            mounts = container.attrs.get("Mounts", [])
            mounted_host_path = next((m["Source"] for m in mounts if m["Destination"] == self.container_internal_data_dir), None)

            if mounted_host_path and os.path.abspath(mounted_host_path) != os.path.abspath(self.host_workspace):
                container.remove(force=True)
                raise NotFound("Recreating due to stale path")

            if container.status != "running":
                container.start()
            return container

        except NotFound:
            return self.client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                tty=True,
                # Mount the HOST path to /data
                volumes={self.host_workspace: {"bind": self.container_internal_data_dir, "mode": "rw"}},
            )

    def analyze_molecule(
        self, molecule: Mol | str, point_groups: list[str], pull_image: bool = False, exact: bool = True, max_conformer_attempts: int = 5000, num_conformers: int = 1, cleanup_on_exit: bool = True
    ) -> MoleculeCSMResult | None:
        container = self._get_or_start_container(pull_image)
        run_id = uuid.uuid4().hex

        # FIX 1: Add prefix to ensure valid filename format
        input_filename = f"mol_{run_id}.sdf"
        local_input_path = os.path.join(self.local_workspace, input_filename)
        paths_to_clean = [local_input_path]

        try:
            molecule = compute_conformer(molecule=molecule, save_file=True, max_attempts=max_conformer_attempts, num_conformers=num_conformers, filename=local_input_path)

            if molecule is None or not os.path.exists(local_input_path):
                return None

            # FIX 2: Ensure file is world-readable so Container User can read it
            try:
                os.chmod(local_input_path, 0o644)
            except Exception:
                pass  # Ignore on Windows or if not owner

            analysis = MoleculeCSMResult(smiles=Chem.MolToSmiles(molecule))
            analysis.input_sdf_path = local_input_path

            container_input_path = os.path.join(self.container_internal_data_dir, input_filename)

            # FIX 3: Verify Visibility immediately
            # If this fails, we know the Mount Path is wrong.
            check_code, _ = container.exec_run(f"test -f {container_input_path}")
            if check_code != 0:
                err_msg = (
                    f"MOUNT ERROR: The container cannot find the file at {container_input_path}.\n"
                    f"Python wrote to: {local_input_path}\n"
                    f"Docker mounted:  {self.host_workspace} -> /data\n"
                    "SOLUTION: If you are running inside a container/DevContainer, you MUST set "
                    "the 'CSM_HOST_MOUNT_PATH' environment variable to the actual path on the host machine."
                )
                # Use c2 as a bucket for this critical error so it surfaces immediately
                analysis.error_messages["c2"] = err_msg
                return analysis

            for pg in point_groups:
                output_dirname = f"output_{pg}_{run_id}"
                local_output_path = os.path.join(self.local_workspace, output_dirname)
                container_output_path = os.path.join(self.container_internal_data_dir, output_dirname)
                paths_to_clean.append(local_output_path)

                command = [
                    "csm",
                    "exact" if exact else "approx",
                    pg,
                    "--input",
                    container_input_path,
                    "--output",
                    container_output_path,
                    "--keep-structure",
                ]

                exit_code, (stdout, stderr) = container.exec_run(command, demux=True)

                if exit_code != 0:
                    analysis.error_messages[pg] = stderr.decode("utf-8") if stderr else "Error"
                    continue

                try:
                    csm_txt_path = os.path.join(local_output_path, "csm.txt")
                    with open(csm_txt_path, "r") as f:
                        data_line = next(line for line in f if not line.startswith("#"))
                        csm_value = float(data_line.strip().split()[-1])
                    analysis.csm_results[pg] = csm_value
                    analysis.output_dirs[pg] = local_output_path
                except Exception as e:
                    analysis.error_messages[pg] = f"Parse error: {e}"

            return analysis

        finally:
            if cleanup_on_exit:
                for path in paths_to_clean:
                    if not os.path.exists(path):
                        continue
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    except PermissionError:
                        # Permission Denied? Ask Docker to delete it.
                        rel_name = os.path.basename(path)
                        container_target = os.path.join(self.container_internal_data_dir, rel_name)
                        try:
                            container.exec_run(["rm", "-rf", container_target])
                        except Exception as e:
                            print(f"Failed to clean up {container_target} via Docker: {e}")
