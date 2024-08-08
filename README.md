
# Flux Image Generation

This repository contains a Python script that uses the `DiffusionPipeline` from the `diffusers` library to generate images based on a provided prompt. The generated images, along with runtime and system information, are logged to a CSV file.  

***This is NOT using GPU, 100% CPU generation (yes, much slower, but works) to test on different Macbook setups I had, but should work for PC CPU too.

## Features

- Generates images using the `DiffusionPipeline` model.
- Uses a provided prompt or generates a random prompt.
- Logs runtime metrics and system information in a `CSV` file.

## Prerequisites

- Python 3.7 or higher

## Installation

### Clone the Repository

```bash
gh repo clone black-forest-labs/flux
cd flux
```

### Setup and Activate Virtual Environment

For macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
pip install flux
```

## Usage

### Run the Script

```bash
python flux_mac_cpu.py
```

### Global Variables

You can tweak the script settings using the global variables defined at the top of the script:

- `PROMPT`: The prompt for image generation.
- `MODEL_ID`: The model ID from the `diffusers` library.
- `NUMBER_OF_INFERENCE_STEPS`: Number of inference steps.
- `CSV_FILENAME`: The name of the CSV file to log runtime metrics.
- `SEED`: The seed for image generation.
- `RANDOMIZE_SEED`: If set to `True`, randomizes the seed value.

Example:

```python
# GLOBAL VARIABLES #
PROMPT = "A curious white cat with striking blue eyes holds a detailed photo of Deadpool..."
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
NUMBER_OF_INFERENCE_STEPS = 1
CSV_FILENAME = "runtime_logs.csv"
SEED = 42
RANDOMIZE_SEED = False
```

### Output

- Generated images are saved in the `generated_images` directory.
- Runtime metrics and system information are logged in the `CSV_FILENAME` file (default: `runtime_logs.csv`).

### CSV Content

The CSV file will contain the following fields:

- `timestamp`
- `create_directories_runtime`
- `load_pipeline_runtime`
- `generate_image_runtime`
- `save_image_runtime`
- `total_runtime`
- `prompt`
- `num_inference_steps`
- `model_id`
- `seed`
- `os`
- `os_version`
- `cpu`
- `cpu_cores`
- `cpu_threads`
- `ram`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Instructions:

1. **Clone the repository** using `gh repo clone black-forest-labs/flux`.
2. **Setup and activate a virtual environment** using provided commands for your respective operating system.
3. **Install required packages** using the `pip install -r requirements.txt` command.
4. **Run the script** with `python flux_mac_cpu.py`.