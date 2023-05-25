# baseline

## Getting started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Optional** (you need a NVIDIA graphic card for this to work):
[Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
[Install TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

## Adding new packages
```bash
pip install <package-name>
pip freeze > requirements.txt
```

## Running code
Work with VSCode. All settings necessary are in `.vscode/settings.json`. 

We use the following syntax for imports to work. Bash: 
```bash
python3 -m folder_name.file_name
```
To debug with breakpoints, you can add a debug config for your file

## Notes
Always use absolute file paths. 