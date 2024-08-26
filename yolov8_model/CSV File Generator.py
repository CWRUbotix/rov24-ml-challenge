import torch
from pathlib import Path
import cv2
from PIL import Image
import pandas as pd
from yolo_wrapper import YoloWrapper

def normalize_data(tensor, image_width, image_height):
    tensor_data_normalized = tensor.clone()
    tensor_data_normalized[0] = tensor[0] / image_width  # Normalize x
    tensor_data_normalized[1] = tensor[1] / image_height  # Normalize y
    tensor_data_normalized[2] = tensor[2] / image_width  # Normalize width
    tensor_data_normalized[3] = tensor[3] / image_height  # Normalize height
    return tensor_data_normalized

def process_video(video_path, output_csv_path):
    torch.set_num_threads(5)  # Adjust the number of threads according to your available resources
    
    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize YOLO model
    model = YoloWrapper('mission_specific.pt')
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    all_frame_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Convert frame to PIL Image for YOLO model
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_width, image_height = image.size
        
        # Make predictions
        data = model.predict_and_show(image)
        
        # Process each detection
        for tensor in data:
            tensor_data_normalized = normalize_data(tensor, image_width, image_height)
            x1 = tensor_data_normalized[0].item()
            y1 = tensor_data_normalized[1].item()
            x2 = tensor_data_normalized[2].item()
            y2 = tensor_data_normalized[3].item()
            # Append frame and bounding box data
            all_frame_data.append([f'Frame {frame_count}', x1, x2, y1, y2])
    
    cap.release()
    
    # Write to CSV file
    df = pd.DataFrame(all_frame_data, columns=["Frame", "x1", "x2", "y1", "y2"])
    df.to_csv(output_csv_path, index=False)

    print(f"Formatted data written to {output_csv_path}")

if __name__ == '__main__':
    video_path = Path('input_video.mp4')  # Replace with your video file path
    output_csv_path = 'output_predictions.csv'
    process_video(video_path, output_csv_path)