# From drugs to materials: Improving knowledge transfer methods for data-scarce materials discovery

Repository for the paper **"From drugs to materials: Improving knowledge transfer methods for data-scarce materials discovery"** — a benchmark of knowledge transfer strategies (pre-training, fine-tuning, reinforcement learning) across three generative AI architectures (REINVENT4, MolMIM+MOLRL, Mol-AIR) for discovering covalent triazine framework (CTF) substrates for supercapacitor electrodes.

---

## Setup

### 0. Clone the repository

```bash
git clone --recurse-submodules <repo-url>
# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

### 1. Python environment

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --build
```

This installs all dependencies including REINVENT4 (as an editable local dependency via `modules/REINVENT4/`) and Mol-AIR (via `modules/mol_air/`).

### 2. Pre-commit hooks

```bash
pre-commit install
```

Hooks will run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

### 3. BioNeMo / MolMIM + MOLRL

The MolMIM+MOLRL model runs inside an NVIDIA BioNeMo Docker container. Requires an NVIDIA GPU and an NGC account.

**Prerequisites:**
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NGC CLI API key (get it from [NGC](https://ngc.nvidia.com/setup/api-key))

**Setup:**
1. Create a `.env` file in the repo root:
   ```
   NGC_CLI_API_KEY=<your-api-key>
   NGC_CLI_ORG=<your-org>
   NGC_CLI_TEAM=<your-team>
   ```
2. Start the container:
   ```bash
   docker compose up
   ```
3. Access JupyterLab at `http://localhost:8899`. The container auto-downloads the MolMIM checkpoint on first start.

---

## Data

### Custom CTF dataset

The curated dataset of 31 covalent triazine framework substrates is stored in:

```
data/raw/data_battery_materials_PRISTINE_CORRECTED_15.09.xlsx
```

Processing this file into train/test splits for both molecular representations (substrate and lattice node) is handled by:

```
notebooks/expert_smiles.py    # Marimo notebook
```

This produces SMILES files for REINVENT's fine-tuning and SELFIES files for Mol-AIR's seed molecules:

```
data/raw/substrate/ctf_train.smi    data/raw/substrate/ctf_test.smi
data/raw/node/ctf_train.smi         data/raw/node/ctf_test.smi
```

Filtered versions for ChEMBL35-compatible vocabulary are also generated (`ctf_train_chembl35.smi`, etc.).

### ChEMBL35 (external)

The ChEMBL35 database is used for pre-training all three architectures. It must be downloaded separately:

```bash
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz
tar -xvzf chembl_35_sqlite.tar.gz -C data/
```

The notebook `notebooks/chembl.py` (Marimo) handles:
1. Extracting SMILES from the ChEMBL SQLite database
2. Converting to SELFIES (for Mol-AIR)
3. Filtering by length (500-atom limit for REINVENT, 256-token limit for Mol-AIR/MolMIM)
4. Generating model-specific vocabularies
5. Splitting into train (70%), validation (10%), test (20%)

Output files are placed under `data/chembl_35_sqlite/`:
- `.smi` files — for REINVENT4
- `.slf` files + `vocab.json` — for Mol-AIR
- `.parquet` files — general-purpose

---

## Models & Running Experiments

Three generative architectures are benchmarked, each across multiple knowledge transfer configurations. Pre-trained model checkpoints are provided in `models/` so you can skip directly to sampling or evaluation.

### REINVENT4

REINVENT4 is an RNN (LSTM) trained on SMILES, with policy-based RL for property optimization. All REINVENT experiments are orchestrated via:

```
scripts/reinvent_run_experiments.py
```

**Quick start — run a single experiment:**

```bash
python scripts/reinvent_run_experiments.py --name REINVENT-node-vanilla-ft-rl
```

**Filter by molecule type, prior, and recipe:**

```bash
# All fine-tuning experiments for substrates using the vanilla prior
python scripts/reinvent_run_experiments.py --molecule-type substrate --prior-type vanilla --recipe ft

# All FT+RL+seed experiments for nodes using ChEMBL35 pretrained
python scripts/reinvent_run_experiments.py --molecule-type node --prior-type chembl35 --recipe ft_rl

# Log aggregated metrics to Weights & Biases
python scripts/reinvent_run_experiments.py --molecule-type substrate --prior-type vanilla --recipe ft_rl --log_wandb
```

**Configuration:**

| File/Dir | Purpose |
|----------|---------|
| `configs/reinvent/base.toml` | Shared parameters (device, paths, TL/RL/sampling defaults, metric reference files) |
| `configs/reinvent/templates/tl_template.toml` | Transfer learning config template |
| `configs/reinvent/templates/rl_template.toml` | Reinforcement learning config template |
| `configs/reinvent/templates/sampling_template.toml` | Sampling config template |
| `configs/reinvent/experiments/node/*.toml` | 13 experiment definitions for lattice node molecules |
| `configs/reinvent/experiments/substrate/*.toml` | 13 experiment definitions for substrate molecules |
| `configs/reinvent/pretrain_reinvent.toml` | ChEMBL35 pre-training config (standalone) |

**Pipeline per experiment:** Transfer Learning (optional) → Reinforcement Learning (optional) → Sampling (10 files × 10,000 SMILES) → Metrics calculation.

**Available priors** (defined in `base.toml`):
- `vanilla` — REINVENT's original ChEMBL25-trained prior. Download from [REINVENT4 releases](https://github.com/MolecularAI/REINVENT4/releases) or generate via `modules/REINVENT4/reinvent/runmodes/create_model/create_reinvent.py`. Place at `models/reinvent/reinvent.prior`.
- `chembl35` — Pre-trained on ChEMBL35 by this project (`models/reinvent/pretrained_chembl_35.model`)

**Available recipes:**
- `sampling` — Sample directly from a prior (no training)
- `ft` — Transfer learning only
- `rl` — Reinforcement learning only
- `rl_inception` — RL with inception (seed molecules)
- `ft_rl` — Fine-tuning followed by RL
- `ft_rl_inception` — Fine-tuning followed by RL with inception

**Pre-computed models** are in `models/reinvent/node/` and `models/reinvent/substrate/` for both molecule types, across all configurations. To sample from a pre-trained model without re-training:

```bash
python scripts/reinvent_run_experiments.py --recipe sampling --molecule-type substrate --prior-type vanilla
```

### Mol-AIR

Mol-AIR is an RL framework for target-aware molecule generation using SELFIES, with LSTM-based actor-critic networks and Random Network Distillation for exploration. Run via:

```
scripts/mol_air_wrapper.py
```

**Quick start:**

```bash
# Vanilla Mol-AIR with seed molecules from the CTF dataset
python scripts/mol_air_wrapper.py configs/mol_air/vanilla.yaml \
    --init_selfies_path data/raw/substrate/ctf_train.slf \
    --inference_runs 10

# ChEMBL35 pretrained Mol-AIR
python scripts/mol_air_wrapper.py configs/mol_air/chembl35_pretrained.yaml \
    --init_selfies_path data/raw/substrate/ctf_train.slf \
    --inference_runs 10
```

**Configuration files:**
- `configs/mol_air/vanilla.yaml` — RL from scratch with Mol-AIR's original weights
- `configs/mol_air/chembl35_pretrained.yaml` — Pre-training on ChEMBL35 + RL fine-tuning

Full parameter documentation for each config section is in `configs/mol_air/README.md`.

**Pipeline:** Optional pre-training → RL training → Inference (generates `n_episodes` molecules per run, repeated `--inference_runs` times with different random seeds).

**Pre-computed models:** `models/mol_air/vanilla/pretrained.pt` and `models/mol_air/vanilla/vocab.json`.

### MolMIM + MOLRL (BioNeMo)

MolMIM is a variational auto-encoder with mutual information machine, pre-trained on ZINC-15. MOLRL applies PPO in MolMIM's latent space for targeted generation. **All MolMIM/MOLRL operations require NVIDIA's BioNeMo container** (see Setup step 3) — they cannot run outside it. The full workflow involves four stages, each driven by a Jupyter notebook running inside the container at `http://localhost:8899`:

**Stage 1 — ChEMBL35 pre-training:** Open the notebook below inside the BioNeMo container's JupyterLab:

```
modules/bionemo/data/MolRL/notebooks/molmim_bionemo/molmim_pretraining.ipynb
```

This adapts NVIDIA's [official pre-training example](https://docs.nvidia.com/bionemo-framework/1.10/notebooks/model_training_molmim.html) to use ChEMBL35 data. Output checkpoint: `/workspace/bionemo/data/models/MolMIM_small_chembl35.nemo`

**Stage 2 — Fine-tuning on CTF data:** Open the notebook inside the BioNeMo container's JupyterLab:

```
modules/bionemo/data/MolRL/notebooks/molmim_bionemo/molmim_finetuning.ipynb
```

This loads a pre-trained checkpoint (vanilla or ChEMBL35) and fine-tunes on the CTF dataset. Output checkpoint: `/workspace/bionemo/data/models/MolMIM_chembl35_finetuning_max_steps_200.nemo`

**Stage 3 — Sampling from a trained model:** Open the notebook inside the BioNeMo container's JupyterLab:

```
modules/bionemo/data/MolRL/notebooks/molmim_bionemo/molmim_sampling.ipynb
```

This performs beam-search-perturbate sampling from the fine-tuned MolMIM model, generating 10,000 molecules per repetition from seed molecules in the CTF dataset. Output: CSV files under `data/outputs/`.

**Stage 4 — MOLRL training:** Open the notebook inside the BioNeMo container's JupyterLab:

```
modules/bionemo/data/MolRL/notebooks/molrl/example_usage_bionemo.ipynb
```

This demonstrates the full MOLRL workflow: defining the battery-property reward function (see Filtering Pipeline below), loading a MolMIM checkpoint, creating the actor-critic network, and running PPO training. The `run_molrl.py` script (`modules/bionemo/data/MolRL/scripts/run_molrl.py`) automates this for 10 repetitions, saving the top-1000 scoring molecules per run to `/workspace/bionemo/data/outputs/molrl/`.

The MolRL library code is at `modules/bionemo/data/MolRL/molrl/`. A simpler standalone example (without BioNeMo integration) is at `modules/bionemo/data/MolRL/notebooks/molrl/example_usage.ipynb`.

---

## Sampling-Based Molecular Set Extension

This is the data augmentation method proposed in the paper. It aggregates constraint-satisfying molecules across multiple generative runs and reuses them as extended fine-tuning data, boosting VUCS from ~3.6% to ~47.9% for substrates.

### Workflow

**Step 1 — Generate molecules:** Run experiments across models (as described above). All generated molecules are accumulated.

**Step 2 — Filter:** Extract molecules that pass all domain constraints:

```bash
python scripts/sampling_filtering_comparison.py \
    --files "data/sampling/reinvent/**/*.csv" \
    --output_comparison data/sampling/filtering_comparison.csv \
    --output_filtered data/sampling/filtered_molecules.csv
```

**Step 3 — Retrieve properties from database:**

```bash
python scripts/sampling-based-database-retrieval.py \
    --glob "data/sampling/**/*.csv" \
    --db modules/bionemo/data/mol_db/substrate_properties.db \
    --out substrate-sampling-based-properties.csv
```

This canonicalizes all SMILES, deduplicates, and joins against the SQLite database to fetch pre-computed filter scores.

**Step 4 — Tokenize-check and split:** Use the Marimo notebook `notebooks/sampling-based-set-extension-data-filtering-and-split.py` to:
- Filter molecules by REINVENT token vocabulary compatibility (vanilla vs ChEMBL35)
- Split into train (80%) / test (20%)
- Output `.smi` files ready for fine-tuning

**Step 5 — Re-run transfer learning** on the extended dataset using the best configuration (REINVENT + FT + RL + seed molecules).

---

## Filtering Pipeline

Five domain-specific constraints evaluate whether a generated molecule is a viable CTF candidate:

| Filter | File | Description |
|--------|------|-------------|
| SMARTS (Structure) | `modules/core/filters/smarts_filter.py` | Inclusion: ≥ 2 C≡N groups. Exclusion: 3/4-membered rings, consecutive C-C bonds, directly bonded N atoms |
| Conjugation | `modules/core/filters/conjugation_filter.py` | Path between any non-bonding electron pair with no two consecutive single bonds |
| Steric Hindrance | `modules/core/filters/steric_hindrance_filter.py` | No two N atoms closer than 4.1 Å |
| Flatness | `modules/core/filters/flatness_filter.py` | Average RMSD from best-fit plane (threshold: 4.0 Å for substrates, 5.0 Å for nodes) |
| Symmetry (CSM) | `modules/core/filters/csm_symmetry_filter.py` | Continuous Symmetry Measure ≤ 0.2 nm/atom for C₂, C₃, or C₄ point groups |

The `MoleculeFilter` class (`modules/core/molecule_filter.py`) orchestrates these filters with SQLite-based caching for performance. Properties are computed once and stored in `modules/bionemo/data/mol_db/{substrate,node}_properties.db`.

---

## Evaluation Metrics

After sampling, metrics are computed via:

```bash
python scripts/calculate_metrics.py \
    --generated_smiles "data/sampling/reinvent/substrate/vanilla/ft_rl/*.csv" \
    --reference_smiles data/raw/substrate/ctf_test.csv \
    --training_smiles data/chembl_35_sqlite/chembl_35_train.smi \
    --run_name "example_experiment" \
    --molecule_type substrate \
    --log_wandb
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| Validity | % of syntactically correct SMILES |
| Uniqueness | % of unique molecules among valid ones |
| Novelty | % of generated molecules not in reference/training sets |
| Internal Diversity | 1 − mean pairwise Tanimoto similarity |
| FCD | Fréchet ChemNet Distance to reference set (lower is better) |
| #Circles | Size of the maximally diverse subset (Tanimoto threshold = 0.5) |
| CSR | Constraint Satisfaction Rate — % passing all 5 domain-specific filters |
| VUCS | Validity × Uniqueness × CSR — the practical yield of candidate molecules |

When evaluating multiple files (from repeated sampling), the script computes mean and standard deviation aggregated across runs. Use `--log_wandb` to log results to Weights & Biases.

---

## Repository Structure

```
├── README.md
├── manuscript.pdf
├── pyproject.toml              # Dependencies, uv config, tool settings
├── docker-compose.yaml         # BioNeMo container setup
├── bionemo_init.sh             # BioNeMo container init script
│
├── configs/
│   ├── reinvent/               # REINVENT4 experiment configs
│   │   ├── base.toml           # Shared base parameters
│   │   ├── templates/          # TL, RL, sampling config templates
│   │   ├── experiments/        # Per-experiment configs (node/ and substrate/)
│   │   ├── pretrain_reinvent.toml
│   │   └── preprocess.toml
│   └── mol_air/                # Mol-AIR experiment configs
│       ├── vanilla.yaml
│       ├── chembl35_pretrained.yaml
│       └── README.md           # Full parameter documentation
│
├── data/
│   ├── raw/                    # Original CTF dataset + expert data
│   │   ├── substrate/
│   │   └── node/
│   ├── chembl_35_sqlite/       # ChEMBL35 processed files (external, must be downloaded)
│   ├── processed_dft_features/
│   ├── processed_all_custom_features/
│   └── sampling/               # Generated molecule outputs
│
├── models/                     # Pre-trained and fine-tuned model checkpoints
│   ├── reinvent/               # REINVENT4 priors + trained models
│   │   ├── node/
│   │   └── substrate/
│   └── mol_air/                # Mol-AIR pre-trained weights + vocab
│
├── modules/
│   ├── core/                   # Shared utilities
│   │   ├── enums.py            # MoleculeType, PriorType, Recipe enums
│   │   ├── molecule_filter.py  # Filter orchestrator with DB caching
│   │   ├── filters/            # Domain-specific constraint filters
│   │   ├── features/           # Feature computation (CSM, flatness, symmetry, etc.)
│   │   └── database/           # SQLite database setup
│   ├── generation/             # Molecule generation & evaluation
│   │   ├── reinvent_wrapper.py # REINVENT4 Python wrapper
│   │   ├── property_evaluator.py
│   │   ├── evaluation.py       # MoleculeGenerationEvaluator
│   │   └── reinvent/
│   ├── REINVENT4/              # REINVENT4 submodule
│   ├── mol_air/                # Mol-AIR submodule
│   └── bionemo/                # BioNeMo / MolMIM + MOLRL integration
│       └── data/MolRL/
│           ├── molrl/           # MolRL library (PPO, reward, latent, vocab)
│           ├── scripts/         # run_molrl.py, molmim_sampling.py
│           └── notebooks/       # Pre-training, fine-tuning, sampling, RL notebooks
│
├── scripts/                    # Top-level execution scripts
│   ├── reinvent_run_experiments.py          # REINVENT experiment orchestrator
│   ├── mol_air_wrapper.py                   # Mol-AIR experiment wrapper
│   ├── calculate_metrics.py                 # Metrics computation (with W&B logging)
│   ├── sampling_filtering_comparison.py     # Compare filter results across runs
│   ├── sampling-based-database-retrieval.py # Retrieve properties from SQLite DB
│   ├── sampling_results_to_xlsx.py
│   ├── batch_update_db.py
│   └── split_excel.py
│
├── notebooks/                  # Marimo notebooks
│   ├── expert_smiles.py        # CTF data parsing & splitting
│   ├── chembl.py               # ChEMBL35 download, conversion, splitting
│   ├── sampling-based-set-extension-data-filtering-and-split.py  # Set extension pipeline
│   ├── chembl_filtering_vis.py
│   ├── circles_comp.py
│   ├── molair.py
│   ├── wandb_plots.py
│   └── ...
│
├── tests/                      # Test suite
├── static/                     # Images and static assets
└── .github/workflows/          # CI configuration
```

---

## Citation

```
[to be added]
```
