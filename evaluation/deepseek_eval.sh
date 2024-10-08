set -ex

numshot=${1}
flantype=${2} 
model_path=${3} 
sampling=${4}
isds=${5}
dataname=${6}
out_folder=${7}
num_total=1 #${1}
rank=0 #${2}


export HOST_IP=0.0.0.0
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python run_open_ds.py \
--model $model_path \
--shots $numshot \
--sampling ${sampling} \
--stem_flan_type ${flantype} \
--batch_size 1024 \
--dataset ${dataname} \
--model_max_length 2048 \
--print --use_vllm \
--num_total ${num_total} \
--rank ${rank} \
--deepseek_templ ${isds} \
--out_path ${out_folder}
