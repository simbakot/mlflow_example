# mlflow_example

pip install mlflow

git clone https://github.com/simbakot/mlflow_example.git

Download '2017 Train/Val annotations' from http://cocodataset.org/#download

Copy instances_train2017.json and instances_val2017.json in project root

mlflow run -P epochs=10 -P categories=cat,dog -P tracking_uri=http://host:port .
