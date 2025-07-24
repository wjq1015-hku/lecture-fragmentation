# Towards Key Point Identification (KPI) for Lecture Videos

This repository contains the source code and implementation for the paper: [Towards Key Point Identification (KPI) for Lecture Videos: Approaches and Performance Evaluation](https://dl.acm.org/doi/10.1145/3746640).

## The `kpi` Framework

The core of this repository is `kpi`, a unified, extensible, and easy-to-use Python framework designed to facilitate designing, implementing, and experimenting with lecture video fragmentation methods.

Key components:

- `kpi/datasets`: An abstraction layer for heterogeneous lecture datasets, including `MITFLD` and `AVLecture`.
- `kpi/models`: Implementations of various fragmentation models discussed in the paper, such as `TextTiling`, `PySceneDetection`, `TW-FINCH`, and `BiLSTM`.
- `kpi/metrics`: Evaluation metrics like Boundary-based F1-score, Mean-over-Frames (MoF), and Intersection-over-Union (IoU).
- `kpi/experiments`: A module to encapsulate all materials for an experiment, including models, test data, and metrics.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/wjq1015-hku/lecture-fragmentation
    cd lecture-fragmentation
    ```

2.  **Install dependencies:**
    The project dependencies are listed in `pyproject.toml`. Create the virtual env and install them using [uv](https://docs.astral.sh/uv/):

    ```bash
    uv sync
    ```

3.  **Prepare datasets:**

    - MITFLD: Download [the dataset](https://huggingface.co/datasets/robotwang/MITFLD).
    - AVLecture: Refer to [Unsupervised Audio-Visual Lecture Segmentation](https://arxiv.org/abs/2210.16644).

4.  **Set up env variables:**
    Copy `.env.example` to `.env` and set the required environment variables:
    - `OPENAI_URL` and `OPENAI_TOKEN` should be set for OpenAI's API for LLM-based fragmentation.

## Usage

The main entry point for the framework is `kpi.py`, which provides a CLI to run a quick experiment with a single model.

To see available commands and options:

```bash
uv run kpi.py --help
```

To run a quick experiment with the `TextTiling` model on the `MITFLD` dataset:

```bash
uv run kpi.py model text_tiling --dataset mitfld --dataset_path <dataset_path> --k 14 --policy top_k --win_type text --win_size 120 --win_step 20 --feat_type sen_embed
```

To see available models and options:

```bash
uv run kpi.py model --help
uv run kpi.py model text_tiling --help

```

To run a demo experiment with multiple models on the `MITFLD` dataset:

```bash
uv run demo_experiment.py --dataset_path <dataset_path>
```

## Citation

If you use this code or the `MITFLD` dataset in your research, please cite our paper:

```bibtex
@article{wang2025towards,
  title={Towards Key Point Identification (KPI) for Lecture Videos: Approaches and Performance Evaluation},
  author={Wang, Jiaqi and Kwok, Ricky Y-K and Ngai, Edith CH},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2025},
  publisher={ACM New York, NY}
}
```
