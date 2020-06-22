import argparse
import os

import mlflow
import mlflow.keras
import mlflow.exceptions
from services.tracking.remote_server import RemoteTracking
from segmentation_project.train_pipeline import Pipeline



class FlowTraining:

    def __init__(self, tracking_uri, model_name, backbone_name, categories, batch_size, image_size=(256, 256, 3)):
        self.tracking_uri = tracking_uri
        self.remote_server = RemoteTracking(tracking_uri=tracking_uri)
        self.local_experiment_dir = './mlruns'
        self.local_experiment_id = '0'

        self.train_pipeline = Pipeline(model_name=model_name,
                                       backbone_name=backbone_name,
                                       categories=categories,
                                       image_size=image_size,
                                       batch_size=batch_size,
                                       path_to_train_annotation='instances_train2017.json',
                                       path_to_train_images='./dataset/train',
                                       path_to_val_annotation='instances_val2017.json',
                                       path_to_val_images='./dataset/val')

    def log_tags_and_params(self, remote_run_id):
        run_id = self.get_local_run_id()
        mlflow.set_tracking_uri(self.local_experiment_dir)
        run = mlflow.get_run(run_id=run_id)
        params = run.data.params
        tags = run.data.tags
        self.remote_server.set_tags(remote_run_id, tags)
        self.remote_server.log_params(remote_run_id, params)

    def get_local_run_id(self):
        files = os.listdir(os.path.join(self.local_experiment_dir, self.local_experiment_id))
        for file in files:
            if not file.endswith('.yaml'):
                return file

    def run(self, epochs, lr, experiment_name):
        # getting the id of the experiment, creating an experiment in its absence
        remote_experiment_id = self.remote_server.get_experiment_id(name=experiment_name)
        # creating a "run" and getting its id
        remote_run_id = self.remote_server.get_run_id(remote_experiment_id)

        # indicate that we want to save the results on a remote server
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=remote_run_id, nested=False):
            mlflow.keras.autolog()
            self.train_pipeline.train(lr=lr, epochs=epochs)

        try:
            self.log_tags_and_params(remote_run_id)
        except mlflow.exceptions.RestException as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', help='list of categories from coco dataset')

    parser.add_argument('--epochs', type=int, help='number of epochs in training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='Unet', help='Unet, PSPNet, Linknet, FPN')
    parser.add_argument('--backbone_name', type=str, default='resnet18', help='exampe resnet18, resnet50, mobilenetv2 ...')

    parser.add_argument('--tracking_uri', type=str, help='the server address')
    parser.add_argument('--experiment_name', type=str, help='remote and local experiment name')

    args = parser.parse_args()

    #  preparing arguments
    categories = args.categories.split(',')

    flow_training = FlowTraining(tracking_uri=args.tracking_uri, model_name=args.model_name,
                                 backbone_name=args.backbone_name, categories=categories, batch_size=args.batch_size)

    flow_training.run(epochs=args.epochs, lr=args.lr, experiment_name=args.experiment_name)