import subprocess
import argparse
import yaml
import os
import json
import logging
import shutil

import pandas as pd

from datetime import datetime
from ultralytics import YOLO

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename=os.path.join(os.getcwd(), 'logs', 'std_' + str(
    datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.log'),
                    format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_performance(path):
    dirs_with_date = []
    dirnames = os.listdir(path)
    for directory in dirnames:
        full_dir_path = os.path.join(path, directory)
        timestamp = os.path.getmtime(full_dir_path)
        modified_date = datetime.fromtimestamp(timestamp)
        dirs_with_date.append((full_dir_path, modified_date))
    return dirs_with_date


def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data,
                                     flow_style=True)


data_path = 'membrane_datasets'

yaml.add_representer(list, represent_inline_list)

parser = argparse.ArgumentParser(
    description='Run a YOLO training task for Classification.')
parser.add_argument('--path', type=str, required=True,
                    help='path for dir')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch', type=int, default=8, help='batch size')
parser.add_argument('--mode', type=str, default='classification',
                    help='type of mode ( e.g classification )')
parser.add_argument('--save', type=str, required=True,
                    help='provide location to save the model')
parser.add_argument('--name', type=str, required=True,
                    help='provide custome name of the model')

args = parser.parse_args()
save_dir = args.save
name = args.name
path = args.path

directory_path = os.path.join(os.getcwd(), save_dir, name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"directory '{directory_path}' created successfully.")
else:
    print(f"directory '{directory_path}' already exists.")
    shutil.rmtree(directory_path)

# labels = args.labels
# labels = labels.split(',')

if args.mode == 'classification':
    # dataset = {
    #     'train': args.train_path,
    #     'val': args.val_path,
    #     'test': args.test_path,
    #     'nc': len(labels),
    #     'names': labels
    # }
    #
    # with open('dataset.yaml', 'w') as file:
    #     yaml.dump(dataset, file, sort_keys=False)

    command = (f"yolo task=classify mode=train data={path} "
               f"model=yolov8m-cls.pt epochs={args.epochs} imgsz=640 "
               f"batch={args.batch} project={args.save} name={args.name} "
               f" amp=True")

    time = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

    with open(os.path.join(os.getcwd(), 'logs', 'training_' + time + '.txt'),
              'w') as output_file:
        process = subprocess.Popen(command, shell=True, stdout=output_file,
                                   stderr=output_file)

    std_out, std_err = process.communicate()

    try:
        res_json = {'status': 'success'}
        if process.returncode == 0:
            fp = os.path.join(save_dir, name, 'results.csv')
            file_exists = os.path.exists(fp)
            if file_exists:
                res = pd.read_csv(fp)
                df = res.iloc[:, 1:7].mean()
                performance_data = {
                    'box_loss': df[0],
                    'seg_loss': df[1],
                    'cls_loss': df[2],
                    'focal_loss': df[3],
                    'precision': df[4],
                    'recall': df[5],

                }
                res_json['performance'] = performance_data
                json_data = json.dumps(res_json)
                print(json_data)
                logger.info(json_data)

            else:
                msg = 'Performance files does not exist'
                res_json['performance'] = msg
                json_data = json.dumps(res_json)
                print(json_data)
                logger.info(json_data)
        else:
            res_json['status'] = 'failure'
            res_json['error'] = std_err
            res_json[
                'performance'] = 'No performance is available due to failure'
            json_data = json.dumps(res_json)
            print(json_data)
            logger.info(json_data)

    except Exception as e:
        print("An error occurred:", str(e))
        logger.info("An error occurred in subprocess:",
                    str(e) + ---------- + std_err)
