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

class_names = [
    'Ballast',
    'Elastomere',
    'EPDM',
    'Multicouche',
    'Revetement metallique',
    'TPO'
]

results = model.predict(resized_image, conf=0.8, iou=0.4, agnostic_nms=True)[0]

for result in results:
    box = result.boxes[0]
    print(box)
    print(box.xyxy)
    x1, y1, x2, y2 = box.xyxy[0]
    class_id = box.cls[0]
    conf = box.conf[0]
    cv.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 0), 2)
    label = f'{class_names[int(class_id)]}: {conf:.2f}'
    cv.putText(resized_image, label, (int(x1) + 5, int(y1) + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv.imwrite(f'{directory_path}/output_image.jpg', resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
