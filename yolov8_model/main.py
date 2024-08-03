from pathlib import Path
from yolo_wrapper import YoloWrapper
from PIL import Image
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
 







