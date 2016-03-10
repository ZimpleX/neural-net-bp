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

current performance:

- `./yaml_model/checkpoint/cell_cnn1.npz`
  - epoch: 20
  - batch: 1500

|           | avg cost | avg accuracy (%) |
| --------- | -------- | ---------------- |
| 0-1500    | 0.371    | 87.133           |
| 1500-3000 | 0.581    | 79.267           |
| 3000-4500 | 0.711    | 76.200           |
| 4500-6000 | 0.626    | 77.067           |
| 6000-7500 | 0.651    | 77.733           |
| Total     | 0.588    | 79.480           |

