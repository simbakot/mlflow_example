from mlflow.tracking import MlflowClient

class RemoteTracking:
    def __init__(self, tracking_uri=None, registry_uri=None):
        self.server = MlflowClient(tracking_uri, registry_uri)

    def get_experiment_id(self, name, artifact_location=None):
        experiment = self.server.get_experiment_by_name(name)
        if experiment:
            experiment_id = experiment.experiment_id
            return experiment_id
        else:
            print("Experiment not exist")
            print("Creating new experiment on tracking")
            experiment_id = self.server.create_experiment(name, artifact_location)
            return experiment_id

    def get_run_id(self, experiment_id):
        run = self.server.create_run(experiment_id)
        run_id = run.info.run_id
        return run_id

    def log_params(self, run_id, params):
        for key, value in params.items():
            self.server.log_param(run_id, key, value)
        print("Parameters successful logged")

    def set_tags(self, run_id, params):
        for key, value in params.items():
            self.server.set_tag(run_id, key, value)
        print("Tags successful logged")

    def log_metrics(self, run_id, params):
        for key, value in params.items():
            self.server.log_metric(run_id, key, value)
        print("Metrics successful logged")

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        self.server.log_artifacts(run_id, local_dir, artifact_path)

