# Recurrent Neural Networks  - A Short TensorFlow Tutorial

### Setup
Clone this repo to your local machine, and add the RNN-Tutorial directory as a system variable to your `~/.profile`. Instructions given for bash shell:

```bash
git clone https://github.com/silicon-valley-data-science/RNN-Tutorial
cd RNN-Tutorial
echo "export PYTHONPATH=$RNN_TUTORIAL/src:${PYTHONPATH}" >> ~/.profile
source ~/.profile
```

If eclipse, remember to add src to pythonpath by right clicking on project,
go to properties, click on PyDev - PYTHONPATH, source folders tab, add source
folder, then choose the src folder.

Create a Conda environment (You will need to [Install Conda](https://conda.io/docs/install/quick.html) first)
Note: This guide helped too (https://conda.io/docs/user-guide/getting-started.html)

```bash
conda create --name tf-rnn
source activate tf-rnn
cd $RNN_TUTORIAL
pip install -r requirements.txt
```

Note: it works with python 3.6 and all new versions of the requirements.

### Install TensorFlow

If you have a NVIDIA GPU with [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) already installed

```bash
pip install --ignore-installed --upgrade tensorflow-gpu 
```
Note: It works with tensorflow 1.5
IMPORTANT!!!: You need to install cuda 9.0 instead of 9.1.
Also install cudnn 9.0 v7, you need to register though.
https://developer.nvidia.com/cudnn

If you will be running TensorFlow on CPU only (i.e. a MacBook Pro), use the following command (if you get an error the first time you run this command read below):

```bash
pip install --upgrade --ignore-installed\
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
```

<sub>**Error note** (if you did not get an error skip this paragraph): Depending on how you installed pip and/or conda, we've seen different outcomes. If you get an error the first time, rerunning it may incorrectly show that it installs without error. Try running with `pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl --ignore-installed`. The `--ignore-installed` flag tells it to reinstall the package. If that still doesn't work, please open an [issue](https://github.com/silicon-valley-data-science/RNN-Tutorial/issues), or you can try to follow the advice [here](https://www.tensorflow.org/install/install_mac).</sub>


### Run unittests
We have included example unittests for the `tf_train_ctc.py` script

```bash
python $RNN_TUTORIAL/src/tests/train_framework/tf_train_ctc_test.py
```


### Run RNN training
All configurations for the RNN training script can be found in `$RNN_TUTORIAL/configs/neural_network.ini`

```bash
python $RNN_TUTORIAL/src/train_framework/tf_train_ctc.py
```

_NOTE: If you have a GPU available, the code will run faster if you set `tf_device = /gpu:0` in `configs/neural_network.ini`_


### TensorBoard configuration
To visualize your results via tensorboard:

```bash
tensorboard --logdir=$RNN_TUTORIAL/models/nn/debug_models/summary/"folder with summary you want"
```

- TensorBoard can be found in your browser at [http://localhost:6006](http://localhost:6006).
- `tf.name_scope` is used to define parts of the network for visualization in TensorBoard. TensorBoard automatically finds any similarly structured network parts, such as identical fully connected layers and groups them in the graph visualization.
- Related to this are the `tf.summary.* methods` that log values of network parts, such as distributions of layer activations or error rate across epochs. These summaries are grouped within the `tf.name_scope`.
- See the official TensorFlow documentation for more details.


### Add data
We have included example data from the [LibriVox corpus](https://librivox.org) in `data/raw/librivox/LibriSpeech/`. The data is separated into folders:

    - Train: train-clean-100-wav (5 examples)
    - Test: test-clean-wav (2 examples)
    - Dev: dev-clean-wav (2 examples)

If you would like to train a performant model, you can add additional wave and txt files to these folders, or create a new folder and update `configs/neural_network.ini` with the folder locations  


### Remove additions

We made a few additions to your `.profile` -- remove those additions if you want, or if you want to keep the system variables, add it to your `.bash_profile` by running:

```bash
echo "source ~/.profile" >> .bash_profile
```

