from pathlib import Path
from yolo_wrapper import YoloWrapper

if __name__ == '__main__':
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
