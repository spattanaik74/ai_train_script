from ultralytics import YOLO

import cv2 as cv
import argparse
import os

parser = argparse.ArgumentParser(description='prediction.')
parser.add_argument('--image_path', type=str, required=True,
                    help='path for training images')
parser.add_argument('--save_path', type=str, required=True,
                    help='path for test images')
parser.add_argument('--model_path', type=str, required=True,
                    help='path for model')
parser.add_argument('--name', type=str, required=True,
                    help='path for test images')

args = parser.parse_args()
image_path = args.image_path
save_path = args.save_path
name = args.name
model_path = args.model_path

directory_path = os.path.join(os.getcwd(), save_path, name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"directory '{directory_path}' created successfully.")

image = cv.imread(image_path)

model = YOLO(f'{model_path}')
h_original, w_original, _ = image.shape

new_width = 640
ratio = new_width / w_original
new_height = int(h_original * ratio)
resized_image = cv.resize(image, (new_width, new_height))


results = model.predict(resized_image, conf=0.8, iou=0.4, agnostic_nms=True)[0]
probs = results.probs

for i, prob in enumerate(probs.data):
    class_name = results.names[i]
    class_prob = prob.item()
    label_text = f"{class_name}: {class_prob:.2f}"
    cv.putText(resized_image, label_text, (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 2)


cv.imwrite(f'{directory_path}/output_image.jpg', resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
