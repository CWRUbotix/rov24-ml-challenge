import torch
from pathlib import Path
from pathlib import Path
from yolo_wrapper import YoloWrapper
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip
# Enable this if you want output video
# __________________________________________________________________

if __name__ == '__main__':


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
        boxes = boxes_tensor.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Convert float coordinates to integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing green box with thickness 2

        return frame

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
    model = YoloWrapper('best.pt')
    # model.train(config_path, epochs=200, name='blood_cell')


    input_video_path = "input_video.mp4"
    output_video_path = "output_video3.mp4"

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
        
        tensor_data = model.predict_and_show(frame)

        frame = draw_bounding_boxes(frame, tensor_data)
        frame_idx += 1
        # Add the processed frame to the list
        processed_frames.append(frame)

    cap.release()
    
    # Convert processed frames to a video
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
    clip.write_videofile(output_video_path, codec="libx264", fps=30)

#___________________________________________________________________

# Enable this if you want output text files
#___________________________________________________________________
if __name__ == '__main__':
# Training
#---------------------------------------------------------------------------------------------

# Comment this out if predicting 
    # paths to the data
    dataset_path = Path('yolo_dataset')  # where the YOLO dataset will be
    large_field_images_path = Path('image')  # where the original images
    labels_path = Path('labels')  # where the labels are


     # create the dataset in the format of YOLO
    YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
    # create YOLO configuration file
    config_path = 'brittle_star_config.yaml'
    YoloWrapper.create_config_file(dataset_path, ['brittle_star'], config_path)

    # create pretrained YOLO model and train it using transfer learning
    model = YoloWrapper('nano')
    model.train(config_path, epochs=200, name='brittle_star')




# Predicting 
#---------------------------------------------------------------------------------------------
# Comment out when purely training 

    dataset_path = Path('yolo_dataset')  # where the YOLO dataset will be
    
    # Switch the model here, make sure to double check the model (Inside the runs folder). 
    model = YoloWrapper('best.pt')


    # make predictions on the validation set
    data_to_predict_path = dataset_path/'images'/'val'

    val_image_list = list(data_to_predict_path.glob('*.png'))
    
    tensor_data = model.predict_and_show(val_image_list[0])

    image = Image.open(val_image_list[0])
    image_width, image_height = image.size

    # Normalize the tensor values based on the image dimensions
    tensor_data_normalized = tensor_data.clone()
    tensor_data_normalized[:, 0] = tensor_data[:, 0] / image_width  # Normalize x
    tensor_data_normalized[:, 1] = tensor_data[:, 1] / image_height  # Normalize y
    tensor_data_normalized[:, 2] = tensor_data[:, 2] / image_width  # Normalize width
    tensor_data_normalized[:, 3] = tensor_data[:, 3] / image_height  # Normalize height

    # Format the normalized values as "x,y,w,h" strings
    formatted_data = ["0 {:.4f} {:.4f} {:.4f} {:.4f}".format(*row) for row in tensor_data_normalized]

    # Write the formatted values to a text file
    with open("image1.txt", "w") as f:
        for line in formatted_data:
            f.write(line + "\n")