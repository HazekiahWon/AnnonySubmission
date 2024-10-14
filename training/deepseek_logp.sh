set -x
suffix=r0
modelname=modelname
modelpath=/path/to/model
datapath=/path/to/data
ds=0
round=0
gpu=8
bash run_logp.sh ${modelpath} ${datapath} ${ds} ${round} ${gpu}
