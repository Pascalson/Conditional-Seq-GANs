# Condition Sequence Generators
This source includes stepwise evaluation methods for sequential GANs, e.g., SeqGAN.
This project is maintained by [Yi-Lin Tuan](https://github.com/Pascalson).


## Requirements
* Developed in Tensorflow r1.3. have also been tested on r1.8
* python 3.6
* library
    * `pip3 install numpy argparse`
* modify the path assigned in `run.sh`

## Usage
We provide the code, script, and [our processed dataset (OpenSubtitles, Counting)](https://github.com/Pascalson/chatbot-data). Feel free to cope with any dataset you prepare, but please make sure the format is consistent with our assigned format.
* The dataset should be placed under directory `./data/`, e.g., `./data/opensubtitles/opensubtitles.txt`. Or you have to check the datapath in the codes.

To run the experiments with default parameters:
```
$bash run.sh <GPU_ID> <TEST_TYPE> <DATA_NAME>
```
- The argument `<GPU_ID>` is used when you have multiple gpu in your computer. You can use command `nvidia-smi` to check the ID of your GPUs. Otherwise, you can just use `0` for `<GPU_ID>`.
- The argument `<TEST_TYPE>` includes `None`, `test_print`, `realtime_argmax`, `realtime_sample`, `realtime_MMI`, `test_D`, etc. All types can be found by `$grep test_type *.py`. Currently, some types are still in maintaining, we do not ensure every function could work. If you have any proplem, contact us, and we will revise it soon.
- The argument `<DATA_NAME>` is the subdirectory name you store your dataset under directory `data/`.

You can change all the parameters in `run.sh` or refer to `args.py`.

## Dependency
training / testing criterions:
* `main.py`
* `args.py`
* `train_mle.py`: MLE
* `train_gan_n_rl.py`: GAN and RL
* `train_others.py`: only discriminator or value network
* `train_utils.py`
* `test.py`: testing
* `test_others.py`: test discriminator

models:
* `seq2seq_model_comp.py`: the main models
* `critic.py`: discriminator modules and critics of synthetic experiments
* `units.py`: basic encoding and decoding

data processing:
* `data_utils.py`
