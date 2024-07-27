
from datetime import datetime
from pathlib import Path
import json
import os
import requests
from PIL import Image
from io import BytesIO
from fathomnet.api import images, boundingboxes


def download_and_process_images(images_folder: str | Path, labels_folder: str | Path) -> None:
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)

    # Create folders if they do not exist
    images_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)
    # Get a list of images for that concept from FathomNet
    random_concept_images = images.find_by_concept('Ophiuroidea')
    
    index = 0
    
    for image_record in random_concept_images:
        index += 1
        response = requests.get(vars(image_record)['url'])
        img = Image.open(BytesIO(response.content))

        # Save the image
        img_name = f"{index}.png"
        img_path = images_folder / img_name
        img.save(img_path)

        # Get image dimensions
        image_width, image_height = img.size

        # Prepare label data
        label_data = []
        for boundingBox in vars(image_record)['boundingBoxes']:
            if vars(boundingBox)['concept'] != 'Ophiuroidea':
                continue

            bbox = vars(boundingBox)

            class_id = 0
            x_left = bbox['x']
            y_top = bbox['y']
            x_right = x_left + bbox['width']
            y_bottom = y_top + bbox['height']
            w = x_right - x_left
            h = y_bottom - y_top



            normalized_x_center = ((x_left + x_right) / 2) / image_width
            normalized_y_center = ((y_top + y_bottom) / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Format the label
            text = (f'{class_id} '
                    f'{normalized_x_center} '
                    f'{normalized_y_center} '
                    f'{normalized_width} '
                    f'{normalized_height}')
            label_data.append(text)

        label_file_path = labels_folder / f"{index}.txt"
        with open(label_file_path, 'w') as file:
            file.write('\n'.join(label_data))




images_folder = 'image'
labels_folder = 'labels'

# Step 1: Download images and process labels
download_and_process_images(images_folder, labels_folder)
print("Finished")

