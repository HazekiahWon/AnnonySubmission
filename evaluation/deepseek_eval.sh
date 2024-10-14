set -ex

numshot=${1}
flantype=${2} 
model_path=${3} 
sampling=${4}
isds=${5}
dataname=${6}
out_folder=${7}
num_total=${9:-'1'}
rank=${10:-'0'}
datapath=${11:-''}
logp=${8}


export HOST_IP=0.0.0.0
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

/home/ma-user/.conda/envs/mammoth/bin/python run_open_ds.py \
--model $model_path \
--shots $numshot \
--sampling ${sampling} \
--stem_flan_type ${flantype} \
--batch_size 1024 \
--dataset ${dataname} \
--data_path ${datapath} \
--model_max_length 2048 \
--print --use_vllm \
--num_total ${num_total} \
--rank ${rank} \
--deepseek_templ ${isds} \
--out_path ${out_folder} \
--use_logp ${logp} 
