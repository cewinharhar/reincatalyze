# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

G-ReInCATALiZE is a **GPU-accelerated Reinforcement learning-enabled Combination of Automated Transformer-based Approaches with Ligand binding and 3D prediction for Enzyme evolution**. The system uses reinforcement learning and transformer models (ESM2) to optimize enzyme mutations for improved catalytic activity with specific ligands.

## Architecture

The codebase follows a modular architecture with specialized helper modules:

- **`src/main.py`** - Main pipeline orchestrator that coordinates all components
- **`src/main_gaesp.py`** - Core GAESP (Genetic Algorithm Evolution with Structure Prediction) execution
- **`src/deepMutHelpers/`** - ESM2 transformer model integration for mutation effect prediction
- **`src/gaespHelpers/`** - Molecular docking utilities and scoring functions (Vina-GPU integration)
- **`src/mainHelpers/`** - Ligand preparation, CAS number processing, and API request utilities
- **`src/residoraHelpers/`** - PPO reinforcement learning components (ActorCritic, ConvNet)
- **`src/pyroprolexHelpers/`** - PyRosetta integration for protein structure manipulation
- **`src/mutantClass.py`** - Core data structure for representing protein mutants
- **`src/configObj.py`** - Configuration management system

## Development Commands

### Running the Pipeline
```bash
# Standard execution
PYTHONPATH=. python src/main.py --config src/CONFIG/config_default.yaml

# Debug mode
PYTHONPATH=. python src/main.py --config src/CONFIG/config_debug.yaml
```

### Testing
```bash
# Run tests
pytest tests/

# Specific test
pytest tests/test_process.py
```

### Code Formatting
```bash
# Format code (Black is configured with 79 character line length)
black src/
```

### Docker Development
```bash
# Build container
docker build --platform linux/amd64 -t gaesp .

# Run with GPU support
docker run -d --gpus all --name <container_name> -p 80:80 gaesp
```

### Package Management
```bash
# Install dependencies
poetry install

# Add new dependency
poetry add <package_name>
```

## Configuration System

The system is **configuration-driven** using YAML files in `src/CONFIG/`:

- **`config_default.yaml`** - Standard production configuration
- **`config_debug.yaml`** - Debug/development settings
- **`config_*.yaml`** - Various experimental configurations

Key configuration sections:
- `globalConfig` - Transformer models, device settings, file paths
- `gaespConfig` - Wildtype sequence, structure paths, docking parameters
- `pyroprolexConfig` - PyRosetta mutation settings

## Key Dependencies

- **PyTorch** - Neural network backbone
- **Transformers (Hugging Face)** - ESM2 protein language model
- **PyMOL** - Protein structure manipulation
- **RDKit** - Chemical structure processing
- **Pandas/NumPy** - Data processing
- **PyRosetta** - Protein design (containerized)
- **Vina-GPU** - GPU-accelerated molecular docking
- **Prefect** - Workflow orchestration

## Development Notes

- The system requires GPU acceleration for both ML models and molecular docking
- ESM2 models are loaded on `cuda:0` by default
- Docker containers include complex molecular biology software stack (Boost, OpenBabel, ADFRsuite)
- Mutation candidates are generated locally around binding sites (configurable distance threshold)
- Pipeline supports both single mutations and combination approaches
- Results visualization available through `GReincatalyze_resultOverview_plot.py`

## File Structure Patterns

- Configuration files use `.yaml` extension
- Helper modules organized by functional area (`*Helpers/`)
- Main execution scripts prefixed with `main_`
- Test files in `tests/` directory follow `test_*.py` pattern
- Docker files in dedicated `docker/` directory with GPU-specific scripts