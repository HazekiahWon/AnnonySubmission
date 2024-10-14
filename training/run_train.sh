set -ex
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
out_base=out
ENV="codellama"
phase=${phase:-''} 
logp_path=${logp_path:-'None'}
rw_path=${rw_path:-'None'}
ispretrain=${ispretrain:-'0'}
numepoch=${numepoch:-'1'}
isweighted=${isweighted:-'3'}
isds=${isds:-'0'}
lr=${lr:-'20'}
round=${round:-'0'}
logp=${logp:-'0'}
use_rho=${rho:-'2'}
kl=${kl:-'1'}
memo=${memo:-''}
tag="r${round}_lr${lr}_w${isweighted}_E${numepoch}_rho${use_rho}_kl${kl}_${memo}"

if [ $phase = "mistral" ]; then 
export Model_layer="MistralDecoderLayer"
else 
export Model_layer="LlamaDecoderLayer"
fi 

if [ $phase = "mistral" ] || [ $phase = "llama3" ]; then 
maxlength=1500
else 
maxlength=2000
fi
ml=${ml:-"${maxlength}"}

modelname="${tag}_E${numepoch}_pre${ispretrain}_w${isweighted}_ds${isds}"
export MODEL_PATH=${mp}
export OUTPUT_PATH="${out_base}/${modelname}"

if [ $ENV = codellama ]; then
export Model_layer="LlamaDecoderLayer"
elif [ $ENV = mistral ]; then
export Model_layer="MistralDecoderLayer"
elif [ $ENV = deepseek ]; then
export Model_layer="LlamaDecoderLayer"
fi

if [ "$tag" == *debug* ]; then # [[ "$fp" == *after* ]] || 
    per_device_train_batch_size=2
    gradient_accumulation_steps=1
elif [ $isweighted == 1 ]; then # [[ "$fp" == *after* ]] || 
    per_device_train_batch_size=1
    gradient_accumulation_steps=16
elif [[ "$MODEL_PATH" == *7B* ]] || [[ "$MODEL_PATH" == *7b* ]]; then
    per_device_train_batch_size=2
    gradient_accumulation_steps=8
else
    per_device_train_batch_size=2
    gradient_accumulation_steps=8
fi

bsz=${bsz:-"${per_device_train_batch_size}"}
gacc=${gacc:-"${gradient_accumulation_steps}"}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export MASTER_ADDR=localhost
export MASTER_PORT="6066"
export WANDB_PROJECT="Math_model"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export OMP_NUM_THREADS=128

NODE_RANK=0
export NUM_GPUS=${gpu}
export WORKER_NUM=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3930
MASTER_PORT="6066"
flash_attn=False
torchrun=/path/to/bin/torchrun
env="dlc"
if [ $NUM_GPUS -gt 1 ]; then
echo "dlc"
export NCCL_NET=IB
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15

MASTER_HOST="$VC_WORKER_HOSTS"
torchrun=/path/to/bin/torchrun
IFS=',' read -r -a array <<< "${VC_WORKER_HOSTS}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
WORKER_NUM="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NUM_GPUS="$MA_NUM_GPUS"
flash_attn=False
fi

echo -e "MASTER_HOST:$MASTER_HOST\nMASTER_ADDR:$MASTER_ADDR\nNODE_RANK:$NODE_RANK\nWORKER_NUM:$WORKER_NUM\n"
echo -e "NUM_GPUS:$NUM_GPUS\nMODEL_PATH:$MODEL_PATH\nper_device_train_batch_size=$bsz\ngradient_accumulation_steps=$gacc" 
$torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  --nnodes=${WORKER_NUM} \
  --master_addr=${MASTER_ADDR} \
  --node_rank=${NODE_RANK} \
  train.py \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length ${ml} \
    --run_name ${modelname} \
    --memo ${memo} \
    --deepseek_templ ${isds} \
    --data_path ${fp} \
    --data_config ${fp} \
    --logp_path ${logp_path} \
    --rw_path ${rw_path} \
    --is_pretrain ${ispretrain} \
    --is_weighted_loss ${isweighted} \
    --bf16 True \
    --output_dir ${OUTPUT_PATH} \
    --num_train_epochs 1 \
    --dataset_repeat ${numepoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gacc} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 3 \
    --round_id ${round} \
    --learning_rate "${lr}e-6" \
    --do_logp ${logp} \
    --use_rho ${use_rho} \
    --kl_weight ${kl} \
    --end_lr 6e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_safetensors True \
    --flash_attn ${flash_attn} \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" 