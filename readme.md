# Supporting Materials for Reproducibility of Submission 5899
## Installation
`pip install -r requirements.txt`

## Training Data
We release the collected dataset from publicly available math instruction-tuning datasets. This collected dataset: 
- aims to **enable coding behaviors** while improving the CoT ability for math queries; 
- contains a comprehensive set of 119,274 **unique queries**
- provides extracted **golden answer** that eases the verification of predicted solutions

Check the `seed_data` folder for details. 

## Training Scripts 
We provide the training scripts in the `training` folder. Check the readme for details.

## Evaluation Scripts
We provide the vllm inference and evaluation scripts in the `evaluation` folder. Check the readme for details.

## Evaluation Results
We provide inference results and detailed matching logs of the trained models. Please see `evaluation_results` folder, and check the readme for details.



## ToDo
- [x] evaluation results of the trained models 
- [x] annonymize evaluation scripts 
- [x] annonymize the collected query set
- [ ] annonymize and publicize trained models
- [x] annonymize training scripts: logp, offpolicy-pg
