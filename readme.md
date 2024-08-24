# MATE ROV 2024 Computer Vision Challenge

CWRUbotix's solution to the MATE ROV 2024 [Ocean Exploration Video Challenge](https://20693798.fs1.hubspotusercontent-na1.net/hubfs/20693798/2024%20OER%20MATE%20ROV%20Computer%20Coding%20Challenge.docx.pdf).

Training footage was provided by MATE [here](https://drive.google.com/file/d/1Wb9GjKUs6-hu4zLdTqaahYo66ZOhCXhr/view).

FathomNet has HuggingFace models [here](https://huggingface.co/FathomNet).

## Running YoloV8

### Install libraries

 - Create & activate a virtual environment in `rov24-ml-challenge`:
   ```bash
   python -m venv env
   . env/bin/activate
   ```
   (the activation is something like `env/Scripts/activate` on Windows)
 - Install the requirements for YoloV8:
   ```bash
   pip install -r yolov8_model/requirements.txt
   ```

### Train
 - Go to `main.py` and double-check that you're in training mode
 - Double-check your images and labels in the respective folders
 - Remove files from previous runs:
  - `runs` (folder)
  - `yolo_dataset` (folder)
  - `brittle_star_config.yaml`
  - `yolov8.pt`
 - `cd yolov8_model`
 - `python main.py`

## Exploding the Video
```bash
mkdir frames
```

Get 1 frame for every 40, from 00m00s to 01m00s:
```bash
ffmpeg -ss 00:00 -i seafloor_footage.mp4 -t 01:00 -vf "select=not(mod(n\,40))" -vsync vfr frames/frame%05d.png
```

Reset:
```bash
rm frames/*
```
