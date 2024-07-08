import datetime
import glob
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2 as cv

import numpy as np
import torch

import gc
# from PIL import Image
#
# import PIL
from ultralytics import YOLO
import statistics

from dask_image.imread import imread

from math import degrees

from skimage.exposure import is_low_contrast

# PIL.Image.MAX_IMAGE_PIXELS = 933120000

def preprocess_image(resize_image_):

    gray = cv.cvtColor(resize_image_, cv.COLOR_BGR2GRAY)

    std_dev = np.std(gray)

    print(f'standard deviation: {std_dev}')

    contrast_threshold = 0

    # if std_dev < contrast_threshold:
    #     contrast_img = cv.convertScaleAbs(image, alpha=1.145, beta=0)
    #     print('contrst low enchancement applied')
    #     return contrast_img
    # else:
    #     return image

    contrast_img = cv.convertScaleAbs(resize_image_, alpha=1.15, beta=0)
    print('contrst low enchancement applied')
    return contrast_img
    # Increase contrast


    # Add shadow effect by darkening the lower part of the image
    # shadow_img = contrast_img.copy()
    # rows, cols, _ = shadow_img.shape
    # for i in range(rows):
    #     alpha = 1 - (i / rows * 0.5)
    #     shadow_img[i, :] = np.clip(shadow_img[i, :] * alpha, 0, 255)

    # cv.imshow('Processed Image', contrast_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()



def predict_roof_resize(model, image_path):
    # Read the image
    image = cv.imread(image_path)
    h_original, w_original, _ = image.shape

    # Resize the image to width 640 pixels while maintaining the aspect ratio
    new_width = 640
    ratio = new_width / image.shape[1]
    new_height = int(image.shape[0] * ratio)
    resized_image = cv.resize(image, (640, 640))

    h, w, _ = resized_image.shape

    # contrast_factor = 1.5  # Adjust as needed
    # contrast_image = cv.convertScaleAbs(resized_image, alpha=contrast_factor,
    #                                      beta=0)

    preprocessed_image = preprocess_image(resized_image)

    results = model.predict(resized_image, conf=0.8, iou=0.4, agnostic_nms=True)[0]

    for result in results:
        masks = result.masks.data
        boxes = result.boxes.data
        clss = boxes[:, 5]

        get_indices = torch.where(clss == 0.0)
        pav_masks = masks[get_indices]
        pav_masks = torch.any(pav_masks, dim=0).int() * 200
        det = pav_masks.cpu().numpy()

        det_binary = np.where(det > 0, 255, 0).astype(np.uint8)

        # extracted_pixels = cv.bitwise_and(resized_image, resized_image,
        #                                   mask=det_binary)
        #
        # gray = cv.cvtColor(extracted_pixels, cv.COLOR_BGR2GRAY)
        #
        # gaussian = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
        # # cv.imshow('gaussian', gaussian)
        #
        # edges = cv.Canny(gaussian, 100, 200)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        dilated = cv.dilate(det_binary, kernel)

        # cv.imshow('edges', edges)

        # Find contours in the edges image
        contours, _ = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

        #
        #
        # cv.imshow('extracted_pixels', extracted_pixels)
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv.contourArea)

        # Draw the largest contour on the original image
        # img_with_contours = cv.cvtColor(resized_image,
        #                                 cv.COLOR_GRAY2BGR) if len(
        #     resized_image.shape) == 2 else resized_image.copy()
        # if largest_contour is not None:
        #     cv.drawContours(resized_image, [largest_contour], -1,
        #                     (0, 255, 0), 2)
        cv.drawContours(resized_image, [largest_contour], -1,
                        (0, 255, 0), 2)

        # cv.imshow('Processed Image', resized_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return resized_image


def straight_roof_detection_line(model, image_path):
    image = cv.imread(image_path, cv.IMREAD_REDUCED_COLOR_2)
    # with Image.open(image_path) as img:
    #     img = img.convert("RGB")
    #     image = np.array(img)

    # image = imread(image_path).compute()
    # image = np.array(image, dtype=np.uint8)

    h_original, w_original, _ = image.shape

    # Resize the image to width 640 pixels while maintaining the aspect ratio
    new_width = 640
    ratio = new_width / image.shape[1]
    new_height = int(image.shape[0] * ratio)
    resized_image = cv.resize(image, (640, 640))

    preprocessed_image = preprocess_image(resized_image)

    results = model.predict(preprocessed_image, conf=0.8, iou=0.2, agnostic_nms=True)[0]

    combined_mask = None

    increment_value = 50

    for result in results:
        masks = result.masks.data
        boxes = result.boxes.data
        clss = boxes[:, 5]

        get_indices = torch.where(clss == 0.0)
        pav_masks = masks[get_indices]
        pav_masks = torch.any(pav_masks, dim=0).int() * 200
        det = pav_masks.cpu().numpy()

        det_binary = np.where(det > 0, 255, 0).astype(np.uint8)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        dilated = cv.dilate(det_binary, kernel)
        blurred = cv.GaussianBlur(dilated, (5, 5), 0)
        if combined_mask is None:
            combined_mask = np.zeros_like(blurred)

        combined_mask = np.where((blurred > 0) & (combined_mask == 0), blurred, combined_mask)

    name = image_path.split('\\')[-1].split('.')[0]
    print(f'image: {name}')



    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

    contour_areas = [cv.contourArea(contour) for contour in contours]

    if contour_areas:
        # Find the median contour area
        median_area = statistics.mean(contour_areas)
        print(f'Median Contour Area: {median_area}')

        list_straigt_contours = []
        # Draw only contours with an area greater than the median area
        for contour in contours:
            print(f'current contour area: {cv.contourArea(contour)}')
            if cv.contourArea(contour) >= 2000:

                epsilon = 0.0075 * cv.arcLength(contour, True)
                straight_contour = cv.approxPolyDP(contour, epsilon, True)
                list_straigt_contours.append(straight_contour)
                cv.drawContours(resized_image, [straight_contour], -1,
                                (0, 255, 0), 2)
    else:
        print('No contours found.')

    cv.destroyAllWindows()
    return resized_image


def main():
    model = YOLO('models/best_100.pt')

    image_path = 'pavement_detection_dataset/test/images'

    images = [file for file in glob.glob(f'{image_path}/*.jpg')]
    print(images)

    save_path = 'test_result/test'

    a = datetime.datetime.now()

    for img in images:
        # print('___image____')
        # img = Image.open(img)
        # width, height = img.size
        #
        # # display width and height
        # print("The height of the image is: ", height)
        # print("The width of the image is: ", width)
        res = straight_roof_detection_line(model, img)
        name = img.split('\\')[-1].split('.')[0]
        print(f'image: {name}')
        cv.imwrite(os.path.join(save_path, name + '_res' + '.PNG'), res)
    b = datetime.datetime.now()

    diff = b - a
    print(f'total time {diff}')

    del model
    del res
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()