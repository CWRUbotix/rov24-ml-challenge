import argparse
import torch
from pathlib import Path
from yolo_wrapper import YoloWrapper
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip


# Enable this if you want output video
# __________________________________________________________________

def draw_bounding_boxes(frame, boxes_tensor):
    """
    Draws bounding boxes on a given frame using YOLO tensor data.

    Parameters:
    frame (numpy.ndarray): The image frame where bounding boxes will be drawn.
    boxes_tensor (torch.Tensor): A tensor containing bounding boxes, each represented as [x1, y1, x2, y2].

    Returns:
    numpy.ndarray: The frame with bounding boxes drawn on it.
    """
    # Ensure the tensor is in CPU and convert it to a numpy array
    boxes = boxes_tensor

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert float coordinates to integers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing green box with thickness 2

    return frame


def label_video(input_video_path: str, output_video_path: str) -> None:
    torch.set_num_threads(5)  # Adjust the number of threads according to your available resources
    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # paths to the data
    dataset_path = Path('yolo_dataset')  # where the YOLO dataset will be
    large_field_images_path = Path('image')  # where the original images
    labels_path = Path('labels')  # where the labels are

    # create the dataset in the format of YOLO
    # YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
    # # create YOLO configuration file
    # config_path = 'brittle_star_config.yaml'
    # YoloWrapper.create_config_file(dataset_path, ['brittle_star'], config_path)

    # create pretrained YOLO model and train it using transfer learning
    model = YoloWrapper('mission_specific.pt')
    # model.train(config_path, epochs=200, name='blood_cell')

    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    processed_frames = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data = model.predict_frame(frame)

        frame = draw_bounding_boxes(frame, data)
        frame_idx += 1
        # Add the processed frame to the list
        processed_frames.append(frame)

    cap.release()

    # Convert processed frames to a video
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
    clip.write_videofile(output_video_path, codec="libx264", fps=30)


def predict_image(input_dir: str, output_dir: str) -> None:
    # dataset_path = Path('yolo_dataset')  # where the YOLO dataset will be
    # data_to_predict_path = dataset_path / 'images' / 'val'

    data_to_predict_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    # Switch the model here, make sure to double-check the model
    model = YoloWrapper('mission_specific.pt')

    # make predictions on the validation set
    val_image_list = list(data_to_predict_path.glob('*.png'))

    for i, image_path in enumerate(val_image_list):
        data = model.predict_frame(image_path)

        image = Image.open(image_path)
        image_width, image_height = image.size

        formatted_data = []
        
        for tensor in data:
            tensor_data_normalized = tensor.clone()  # Create a copy of the tensor
            tensor_data_normalized[0] = tensor[0] / image_width  # Normalize x
            tensor_data_normalized[1] = tensor[1] / image_height  # Normalize y
            tensor_data_normalized[2] = tensor[2] / image_width  # Normalize width
            tensor_data_normalized[3] = tensor[3] / image_height  # Normalize height

            # Format the normalized values as "0 x y w h" strings
            formatted_string = "0 {:.4f} {:.4f} {:.4f} {:.4f}".format(*tensor_data_normalized)
            formatted_data.append(formatted_string)

        # Write the formatted values to a text file
        with open(output_dir_path / f'{image_path.name}.txt', 'w') as f:
            for line in formatted_data:
                f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='predict.py',
                                     description='Uses the model to make predictions on an image or video')
    subparsers = parser.add_subparsers(title='mode')

    predict_video_parser = subparsers.add_parser('video', help='Make predictions on each frame of a video')
    predict_video_parser.add_argument('input_video', help='Path to input video file')
    predict_video_parser.add_argument('output_video', help='Where to output the annotated video')

    predict_video_parser = subparsers.add_parser('image', help='Make predictions on a folder of images')

    args = parser.parse_args()
    print(args)
