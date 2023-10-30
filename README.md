# baseline

## Docker

### On the workstations at AIFB
- On our workstations we use rootles docker where you just switch the ```docker``` in the commands to ```podman```.
- Also the ```--gpus all``` needs to be replaced with ```--device=nvidia.com/gpu=all```.
- At the moment it only works on ```udo```. You can access udo through ```ssh <YOUR-U-KUERZEL>@aifb-atks-udo.aifb.kit.edu```
- I don't know if the above works without a screen so maybe better access udo through remote desktop. Look [here](https://gitlab.kit.edu/kit/aifb/ATKS/helpdesk/wiki/-/wikis/Remote-Desktop) for instructions.

### Create the Docker image
```bash
docker build -t emotion_recognition .
```
### Run the built Docker image
```bash
docker run --rm --gpus all --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix emotion_recognition:latest
```


## Getting started without Docker and without ZED
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