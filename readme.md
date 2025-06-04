#  A Sparse Dynamic Graph Transformer for traffic flow prediction

<img src="./framework.png" width="30%">

## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Data

The dataset link is [Google Drive](https://drive.google.com/drive/folders/176Uogr_kty02NQcM9gB2ZT_ngulEhb0H?usp=share_link). You can download the datasets and place them in the `raw_data` directory.

All 7 datasets come from the [LibCity](https://github.com/LibCity/Bigscity-LibCity) repository, which are processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format. The only difference with the datasets provided by origin LibCity repository [here](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing) is that the filename of the datasets are differently.



## Train & Test

You can train and test **SDGFormer** through the following commands for 7 datasets. Parameter configuration (**--config_file**) reads the JSON file in the root directory. If you need to modify the parameter configuration of the model, please modify the corresponding **JSON** file.

```shell
python run_model.py --task traffic_state_pred --model SDGFormer --dataset PeMS03 --config_file PeMS03
python run_model.py --task traffic_state_pred --model SDGFormer --dataset PeMS04 --config_file PeMS04
python run_model.py --task traffic_state_pred --model SDGFormer --dataset PeMS08 --config_file PeMS08
python run_model.py --task traffic_state_pred --model SDGFormer --dataset PeMS07 --config_file PeMS07
python run_model.py --task traffic_state_pred --model SDGFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model SDGFormer --dataset CHIBike --config_file CHIBike --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model SDGFormer --dataset T-Drive --config_file T-Drive --evaluator TrafficStateGridEvaluator
```

If you have trained a model as above and only want to test it, you can set it as follows (taking PeMS08 as an example, assuming the experiment ID during training is $ID):

```shell
python run_model.py --task traffic_state_pred --model SDGFormer --dataset PeMS08 --config_file PeMS08 --train false --exp_id $ID
```

**Note**: By default the result recorded in the experiment log is the average of the first n steps. This is consistent with the paper (configured as **"mode": "average"** in the JSON file). If you need to get the results of each step separately, please modify the configuration of the JSON file to **"mode": "single"**.






