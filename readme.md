# MATE ROV 2024 Computer Vision Challenge

CWRUbotix's solution to the MATE ROV 2024 [Ocean Exploration Video Challenge](https://20693798.fs1.hubspotusercontent-na1.net/hubfs/20693798/2024%20OER%20MATE%20ROV%20Computer%20Coding%20Challenge.docx.pdf).

## Introduction and Discussion
For the 2024 MATE ROV NOAA Ocean Exploration Video Challenge, CWRUbotix has fine-tuned an Ultralytics YOLOv8 Nano model to recognize brittle stars (*Ophiura ophiura*) in underwater ROV footage. YOLOv8 was selected for its ease of integration with Python image processing and broad object recognition capabilities. Our model operates on a frame-by-frame basis, meaning it is equally successful at recognizing brittle stars in still images (i.e., without the context of the preceding video frames), but cannot use previous frames' predictions to inform future video frames. We fine-tuned YOLOv8 with PyTorch, using PyTorch's Cuda integration to leverage Nvidia GPUs. 

We created a dataset of labeled brittle star images by labeling frames of the provided NOAA ROV footage and scraping pre-labeled data from Fathomnet. By building a composite dataset with images from outside the competition footage, we cut down on the amount of time we spent labeling data and reduced the risk of overfitting, making our model more applicable to a variety of prediction footage. We partitioned our dataset with a 90%-to-10% training-to-testing split.

## User Guide
### Installation
We have uploaded both our model file (`.pt`) and a compressed copy of our codebase for the competition submission. Note that these instructions assume our codebase has been downloaded, but that it is possible to use the model file without our codebase in a custom testing environment. An Nvidia GPU is strongly recommended for running this model.

There are two model files present in the codebase, `mission_specific_model.pt` and `wide_application_model.pt`. The former was trained on a dataset with a larger proportion of NOAA ROV footage, and the latter was trained on a dataset with a larger proportion of Fathomnet general-purpose data. The `mission_specific_model.pt` version is recommended, as it is better tailored to this challenge. The mission specific model is the one uploaded separately in the submission.

Before setting up our codebase, install Python 3.11. Then, to install our codebase:
 1. Either
 - download and uncompress our code from the competition submission, or
 - clone our Git repository: `git clone https://github.com/CWRUbotix/rov24-ml-challenge.git`

 2. Navigate to the source code directory, then install the necessary Python package dependencies with Pip:
```bash
pip install -r requirements.txt
```

3. Install PyTorch compiled with CUDA 12.4 from https://pytorch.org/get-started/locally/. Restart your computer.
4. Navigate to the yolov8_model directory and run the following to verify the installation:
```bash
python3 verify_installation.py
```
If you see a message telling you to install CUDA, proceed to step four. If you see a success message, continue to Making Predictions or Training.

5. Make sure you installed PyTorch compiled with CUDA 12.4. If CUDA still is not installed, download and install CUDA 12 from Nvidia: https://developer.Nvidia.com/cuda-downloads

### Making Predictions
Our model can predict brittle star positions in still images or MP4 videos. Be sure to navigate to the yolov8_model directory before running any of these scripts. Both prediction modes support a `-c` option to accept a decimal confidence threshold between 0.0 and 1.0.

#### Images
To output predicted brittle star positions for still images, place your PNG images in a directory (`-i`). Then run:

```bash
python3 predict.py images -i ./folder/of/images/ -o ./folder/for/predictions/
```

When complete, the coordinates of the bounding boxes will be written to text files in the `-o` argument's directory.

#### Videos
To render bounding boxes around brittle star positions in an MP4 video, run:

```bash
python3 predict.py video -i ./path/to/video.mp4 -o ./path/to/dir
```

When complete, a labeled copy of the video will appear in the directory represented by the `-o` path.

### Training
Training the model is not required to test it against new inputs, but it is possible. To train the model, place training images and labels in the `image` and `labels` directories, with the same format as the dataset already present. Delete `runs`, `yolo_dataset`, `brittle_star_config.yaml`, and any model files (if they exist). Then run:

```bash
python3 train.py
```

## Data Collection Notes
Training footage was provided by MATE [here](https://drive.google.com/file/d/1Wb9GjKUs6-hu4zLdTqaahYo66ZOhCXhr/view).

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
