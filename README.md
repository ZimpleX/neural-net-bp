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