import os
import shutil
import argparse


def create_yolo_annotation(data_set_path, labels_path, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    classes = os.listdir(data_set_path)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_set_path, cls)
        for img_file in os.listdir(cls_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(cls_path, img_file)
                annotation_name = os.path.splitext(img_path)[0] + '.txt'
                annotation_path = os.path.join(labels_path, annotation_name)
                print(annotation_path)

                with open(annotation_path, 'w') as f:
                    f.write(f"{class_to_index[cls]} 0.5 0.5 1.0 1.0\n")

                shutil.copy(img_path, image_path)
                shutil.copy(annotation_path, labels_path)

                if os.path.exists(annotation_path):
                    os.remove(annotation_path)
                    print(f"File {annotation_path} has been deleted.")
                else:
                    print(f"File {annotation_path} does not exist.")


parser = argparse.ArgumentParser(description='Annotation Converter')
parser.add_argument('--data_path', type=str, required=True,
                    help='path for training images')

args = parser.parse_args()
data_path = args.data_path

train_image_path = 'train/images'
train_label_path = 'train/labels'
test_image_path = 'test/images'
test_labeL_path = 'test/labels'

data_set_folder_path = f'{data_path}'
dataset_path = [data_set_folder_path + '/train',
                data_set_folder_path + '/test']

folder_name = os.path.basename(os.path.normpath(data_set_folder_path))

for i in range(len(dataset_path)):
    if i == 0:
        create_yolo_annotation(dataset_path[i],
                               folder_name + '/' + train_label_path,
                               folder_name + '/' + train_image_path)
    if i == 1:
        create_yolo_annotation(dataset_path[i],
                               folder_name + '/' + test_labeL_path,
                               folder_name + '/' + test_image_path)
