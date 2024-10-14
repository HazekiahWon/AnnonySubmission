Training includes iterations of data generation -> data logp -> off-policy RL training 
## Data Generation
We first collect meta information from the inferred responses, and then use that information to generate the data. 

1. Creating the meta information 

    We provide an entry bash script for example: `construct_meta.sh`. Custom parameters include:
    - suffix: the round-tag for the output data name 
    - expname: the model-tag of the model that generates the inference results
    - version: the version-tag for the output meta data name. Either "auto" or "trigger" 

    The script will read the inferred responses from `results/`, and compute the meta data file into `meta/{suffix}_auto_q2responses_{expname}.pkl` or `meta/{suffix}_trigger_q2responses_{expname}.pkl`.

2. Creating the data 

    We provide an entry bash script for example: `construct_data.sh`. Custom parameters include:
    - suffix: the round-tag for the output data name 
    - expname: the model-tag of the model that generates the inference results
    - tag: the version-tag for the output data name 

    The script will collect query-response information from the meta file `meta/{suffix}_auto_q2responses_{expname}.pkl` and `meta/{suffix}_trigger_q2responses_{expname}.pkl`, and dump the resulting data into `data/offline_{tag}_{suffix}_{expname}.pkl`.

## Data LogP
We provide an entry bash script for example: `deepseek_logp.sh`. Custom parameters include:
- modelname: a model-tag for the output data name
- modelpath: the model path
- datapath: the data path
- suffix: a round-tag for the output data name
- round: the iteration of RL training
- ds: use the deepseek prompt template (1) or the alpaca template (0)

## RL Training 
We provide an entry bash script for example: `deepseek_train.sh`. Custom parameters include:
- fp: the data file path
- mp: the model path
- logp_path: the logp path
- memo: A tag for the output model
- lr: the learning rate
- round: the iteration of RL training
- isds: use the deepseek prompt template (1) or the alpaca template (0)

The bash scripts are directed to the python script `train.py`, which uses the deepspeed backend. 

Current scripts are in their crude version, some beta features are not removed. We will further clean the training script. 