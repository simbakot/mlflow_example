from mlflow.tracking import MlflowClient

class RemoteRegistry:
    def __init__(self, tracking_uri=None, registry_uri=None):
        self.server = MlflowClient(tracking_uri, registry_uri)

    def create_registered_model(self, name):
        registered_model = self.server.create_registered_model(name)
        return registered_model

    def update_registered_model(self, name, new_name=None, description=None):
        self.server.update_registered_model(name, new_name, description)

    def get_registered_model(self, name):
        registered_model = self.server.get_registered_model(name)
        return registered_model

    def get_model_version_download_uri(self, name, version):
        download_uri = self.server.get_model_version_download_uri(name, version)
        return download_uri

    def download_artifact(self, run_id, path, dst_path=None):
        return self.server.download_artifacts(run_id, path, dst_path)

    def get_last_model(self, registered_model, stage='Production'):
        for model in registered_model.latest_versions:
            if model.current_stage == stage:
                return model


if __name__ == "__main__":
    server = RemoteRegistry(tracking_uri="http://10.22.12.24:8000")
    registered_model = server.get_registered_model("FaceBoxA")
    download_uri = server.get_model_version_download_uri("FaceBoxA", 3)
    import mlflow.tensorflow
    import tensorflow as tf

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    model = mlflow.tensorflow.load_model(download_uri, tf_sess)
    print()