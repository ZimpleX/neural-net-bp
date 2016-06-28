## About

This is a self-implemented version of ImageNet (DCNN). The net includes `fully-connected` layers, and `convolutional` layers (include `pooling`). The net is supposed to be compatible with [Apache Spark](spark.apache.org), so that the training process can be accelerated by the [Amazon EC2](aws.amazon.com/ec2). 

- language version: `python 3`
- py-packages: 
  - `numpy`, `yaml` (required for DCNN)
  - `sqlite3` (optional, for profiling plotter)
  - `pyspark` (optional, for `Spark` acceleration)
  - `subprocess` (optional, for connection with `ec2`)
- net structure:
  - Specify details of each layer as `*.yaml` file
  - `FC` layers:
    - support user-defined net activation function & cost function
  - `convolutional` layers:
    - support any `stride` & `padding`
    - `feed-forward` & `back-propogation` are expressed in a high-level function `slid_win_4d_flip`

## Directory Structure

- `main.py`: takes one `yaml` model to train
- `conv_unittest.py`: for unit-test & profiling the convolution operation ONLY
- `sweep.py`: takes all `yaml` models in a directory for training. Suitable for design space exploration. 
- `net`: ...
- `conv`: ...
- ...

## ImageNet Model Design

- `./yaml_model/checkpoint/cell_cnn1_3000_final_chkpt.npz`
  - epoch: 46
  - batch: 8400
  - 3 conv layer
  - feature map: 32 - 64 - 128

| name       | data size | accuracy (%) |
| ---------- | --------- | ------------ |
| training   | 3000      | 99.9         |
| validation | 1500      | 88.3         |
| testing    | 3000      | 89.1         |



- `./yaml_model/checkpoint/cell_cnn2_4500_final_chkpt.npz`
  - epoch: 40
  - 3 conv layer
  - feature map: 16 - 32 - 64

| name       | data size | accuracy (%) | cost  |
| ---------- | --------- | ------------ | ----- |
| train      | 4500      | 100.000      | 0.002 |
| validation | 1500      | 90.200       | 0.478 |
| testing    | 1500      | 90.200       | 0.460 |


## Yaml Example
```
data_dir: path/to/data		# dir containing the train/valid/test data
data_format: h5 or npz
```