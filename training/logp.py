#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import json
import math
import datasets
import pdb
from glob import glob
import torch
import numpy as np
import transformers
import pickle as pkl
from torch.utils.data import Dataset
from transformers import Trainer, get_scheduler
import pathlib
import utils
import re
import random
import torch.distributed as dist
import wandb
from collections import deque, defaultdict
from trl.core import logprobs_from_logits
import sys
from evaluation.prompt_utils import get_alpaca_format_prompt_wo_input
from transformers import TrainerCallback

def print_on_rank_0(*args):
    if dist.get_rank() == 0:
        print(*args)
        
#from trl import SFTTrainer
#os.system("echo $PYTORCH_CUDA_ALLOC_CONF")
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_Pause_TOKEN = "<pause>"
padding_length = 160


class RewardQueue:
    def __init__(self, windowsize=50, batchsize=1, use_running=False, report_num=32): # batchsize=4x8
        self.maxsize = report_num
        self.batchsize = batchsize
        self.q = deque(maxlen=self.maxsize)
        self.init_mean = 0.
        self.std = 1.0
        self.use_running = use_running
        self.report_num = report_num
        # accelerator.print(f"reward queue use_running={use_running} when normalizing rewards.")
    
    def append(self, entry):
        self.q.append(entry)
        
    def extend(self, items):
        self.q.extend(items)
        
    def stats(self, running=False): # running mean seems to make policy worse
        if len(self.q)>self.report_num//2 and running: 
            return np.mean(self.q), self.std 
        else: return None, self.std

    def normalize_reward(self, r):
        m,s = self.stats(self.use_running)
        return (r-m)/s
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: Optional[str] = field(default="right")
    deepseek_templ: int = field(default=0)
    round_id: int = field(default=0)
    do_logp: int = field(default=0)
    show_kl: int = field(default=1)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_config: str = field(default="")
    template_variation: bool = field(
        default=True, metadata={"help": "whether to use template variation"})
    direct: bool = field(default=False)
    add_pause_to_answer: bool = field(default=False)
    pause: bool = field(default=False)
    is_pretrain: int = field(default=0)
    is_weighted_loss: int = field(default=0)
    logp_path: str = field(default="None", metadata={"help": "Path to the logp data."})
    rw_path: str = field(default="None", metadata={"help": "Path to the reward data."})
    dataset_repeat: float = field(default=1)
    memo: str = field(default="")
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attn: bool = field(default=False)
    run_name: str 
    end_lr: float = field(default=6e-6)
    num_cosine_steps: int = field(default=6000)
    num_warmup_steps: int = field(default=200)
    kl_weight: float = field(default=1.0)
    kl_discount: float = field(default=0.1)
    use_rho: int = field(default=0)
    


transformers.logging.set_verbosity_info()
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    # sources: Sequence[str],
    # targets: Sequence[str],
    data, 
    tokenizer: transformers.PreTrainedTokenizer,
    is_pretrain=False
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources, targets, logps = data
    examples = [s + t for s, t in zip(sources, targets)]
    # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    
    if not is_pretrain:
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    else:
        print_on_rank_0('*'*10, "pretrain")
    return dict(input_ids=input_ids, labels=labels)


def get_data(data_arg, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool, ratio=0):
    config, data_path = data_arg
    use_neg = data_args.is_weighted_loss>=2
    if os.path.exists(config) and 'config' in config:
        data_config = json.load(open(config))
    else: data_config = dict(default=data_path)
    logp_path = data_args.logp_path
    rw_path = data_args.rw_path
    
    all_dat = []
    for k, data_path in data_config.items():
        logging.warning(f"{k}: Loading data from {data_path}")
        if os.path.exists(data_path):
            
            if data_path.endswith(".json"):
                list_data_dict = json.load(open(data_path))
            elif data_path.endswith(".pkl"):
                logp_dict = {}
                if model_args.do_logp==0 and model_args.round_id>=1:
                    # assert os.path.isdir(data_path+'_logp'), f"for round {model_args.round_id}, logp folder {data_path+'_logp'} is required."
                    if logp_path=="None":logp_data_files = glob(data_path+'_logp/*')
                    else: logp_data_files = glob(logp_path+'/*')
                    for fp in logp_data_files:
                        tmpd = pkl.load(open(fp,"rb"))
                        logp_dict.update(tmpd)
                    
                    rw_dict = dict()
                    if rw_path != "None":
                        print('using estimated rewards')
                        
                        rw_data_files = glob(rw_path+'/*')
                        for fp in rw_data_files:
                            tmpd = pkl.load(open(fp,"rb"))
                            rw_dict.update(tmpd)
                        
                list_data_dict = pkl.load(open(data_path,"rb"))
                
                if len(logp_dict)>0:
                    for item in list_data_dict:
                        is_syn = not item['source'].startswith('self')
                        item['reward_type'] = 'self' 
                        item['raw_reward'] = float(not item.get('neg', False))
                        if is_syn: continue 
                        qid = item['qid']
                        logp = logp_dict.get(qid, None)
                        item['logp'] = logp
                        # item['reward_type'] = 'self' # other reward types we don't normalize rewards with them 
                        tmp = rw_dict.get(qid, None)
                         
                        if tmp is not None: 
                            item['reward_type'] = 'estimated'
                            item['estimated_reward'] = tmp[1]
                
                qset = defaultdict(list)
                for idx,item in enumerate(list_data_dict):
                    q = item['instruction']
                    qset[q].append(idx)
                
                for iidx,(q,idxlist) in enumerate(qset.items()):
                    notsyn_rewards = np.asarray([not list_data_dict[idx].get('neg', False) for idx in idxlist if list_data_dict[idx].get('reward_type', None)=='self'], dtype=np.float32)
                    is_ac = sum(notsyn_rewards) == len(notsyn_rewards)
                    is_an = sum(notsyn_rewards) == 0 
                    std = 1.0 if is_ac or is_an else np.std(notsyn_rewards)
                    mean = 0.0 if is_ac or is_an else np.mean(notsyn_rewards)
                    
                    
                    # rewards = []
                    for idx in idxlist: 
                        list_data_dict[idx]['group_idx'] = iidx 
                        r = float(not list_data_dict[idx].get('neg', False)) 
                        if list_data_dict[idx].get('reward_type', None)=='self':
                            r = (r-mean)/std
                            list_data_dict[idx]['reward'] = np.clip(r, -1.0, 1.0) if data_args.is_weighted_loss==3 else r
                        else: 
                            list_data_dict[idx]['reward'] = r
                        
                print(len(list_data_dict), len(logp_dict))
                notsyn_rewards = [item['reward'] for item in list_data_dict if item.get('reward_type')=='self']
                syn_rewards = [item['reward'] for item in list_data_dict if item.get('reward_type') is None]
                print('syn items', len(syn_rewards), 'notsyn items', len(notsyn_rewards))
                if len(notsyn_rewards)>0:
                    print('not syn items reward:', len(notsyn_rewards), np.mean(notsyn_rewards), np.max(notsyn_rewards), np.min(notsyn_rewards))
                if len(syn_rewards)>0:
                    print('syn items reward:', len(syn_rewards), np.mean(syn_rewards), np.max(syn_rewards), np.min(syn_rewards))
                # import pdb; pdb.set_trace()
            else:
                list_data_dict = []
                with open(data_path, 'r') as f:
                    for line in f.readlines():
                        list_data_dict.append(json.loads(line))
            
        else:
            list_data_dict = datasets.load_dataset(data_path)["train"]
        print(f"raw data num {len(list_data_dict)}")
        if not use_neg and model_args.do_logp==0: # we don't drop when doing logp
            list_data_dict = [x for x in list_data_dict if not x.get('neg', False)]
            print(f"neg samples removed, now data num {len(list_data_dict)}")
        all_dat.extend(list_data_dict)
    
    
    print(all_dat[0])
    repeat = data_args.dataset_repeat-1
    force_noshuffle = 'force_noshuffle' in data_args.memo
    if not force_noshuffle:
        new_dat = []
        if repeat>0: 
            for _ in range(int(repeat)):
                new_dat.extend(all_dat)
            all_dat.extend(new_dat)
        elif repeat<0: 
            random.seed(42)
            random.shuffle(all_dat)
            num_keep = int(data_args.dataset_repeat*len(all_dat))
            all_dat = all_dat[:num_keep]
    if force_noshuffle:
        training_args.num_train_epochs = data_args.dataset_repeat
        print('using num_train_epochs for data repeat')
    
    logging.warning("Formatting inputs...")
        
    sources = []
    targets = []
    weights = []
    logps = []
    ids = []
    estimated = []
    outcomes = []
    # import pdb; pdb.set_trace()
    print_on_rank_0('*'*10, 'peek data')
    typeset = set()
    random.seed(42)
    random.shuffle(all_dat)
    a,b = get_alpaca_format_prompt_wo_input([])
    
    for eid, example in enumerate(all_dat):
        if model_args.round_id>0: # in round>0, we use logp, so we must make sure the prompt context aligns
            if 'req' not in example:
                prompt_no_input = prompt_input = a+b.replace("query","instruction")
                sources.append(prompt_no_input.format(instruction=example['instruction']))
            else:
                sources.append(f"{example['req']}")
            targets.append(f"{example['output']}{tokenizer.eos_token}")
            
        else:
            if template_variation:
                PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
            else:
                PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            if model_args.deepseek_templ==1:
                prompt_no_input = prompt_input = "User:{instruction}\n\nAssistant:"
        
            if data_args.direct:
                if 'The answer is' in example['output']:
                    if example.get("input", "") != "":
                        sources.append(prompt_input.format_map(example))
                    else:
                        sources.append(prompt_no_input.format_map(example)) 
                    answer = example['output'].split('The answer is')[-1]
                    answer = f'The answer is{answer}{tokenizer.eos_token}'
                    targets.append(answer)
            else:
                if example.get("input", "") != "":
                    sources.append(prompt_input.format(instruction=example['instruction']))
                else:
                    sources.append(prompt_no_input.format(instruction=example['instruction']))
                    targets.append(f"{example['output']}{tokenizer.eos_token}")
        if example['type'] not in typeset:
            print_on_rank_0(f'type = {example["type"]}')
            print_on_rank_0(sources[-1])
            print_on_rank_0('-'*10)
            print_on_rank_0(targets[-1])
            typeset.add(example['type'])
        # weights.append(1.0/example.get("total_resp", 1.0))
        ispot = 'pot' in example['qid']
        isneg = example.get('neg', False)
        if data_args.is_weighted_loss>=3:
            weights.append(example.get('reward',1.0))
        elif data_args.is_weighted_loss==2:
            weights.append(-1.0 if isneg else 1.0)
        elif data_args.is_weighted_loss==1:
            weights.append(1.5 if ispot else 1.0)
        else:weights.append(1.0)
        logps.append(example.get('logp', None))
        ids.append(example.get('qid', eid))
        estimated.append(example.get('estimated_reward', 1.0))
        outcomes.append(example.get('raw_reward',0))
    
    
    print_on_rank_0("weights mean", np.mean(weights))
    print_on_rank_0("weights max min", np.max(weights), np.min(weights))
  
    num_trn = len(sources)
    
    ratio = int(0.99 * len(sources))
    print_on_rank_0(sources[0])
    print_on_rank_0("*"*30)
    print_on_rank_0(targets[0])
    train_data = [sources[0:num_trn],targets[0:num_trn], weights[:num_trn], logps[:num_trn], ids[:num_trn]]
    eval_data = [sources[num_trn:],targets[num_trn:], weights[num_trn:], logps[num_trn:], ids[num_trn:]]
    
    return train_data,eval_data

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(SupervisedDataset, self).__init__()
        
        self.sources = data[0]
        self.targets = data[1]
        self.weights = data[2]
        self.logps = data[3]
        self.ids = data[4]

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], weights=self.weights[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=(self.targets[i],self.weights[i], self.logps[i], self.ids[i]))

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    is_pretrain: bool
    is_weighted_loss: bool

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, weights = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "weights"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            weights=weights, 
        )
        # if self.is_weighted_loss:
        #     ret['weights'] = weights
        return ret 
        

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        weights = []
        logps = []
        ids = []
        rewards = []
        outcomes = []
        use_estimated_reward = data_args.rw_path != 'None'
        for instance in instances:
            source = instance['input_ids']
            target,weight,logp,qid = instance['labels']
            
            sources.append(source)
            targets.append(target)
            weights.append(weight)
            logps.append(logp)
            ids.append(qid)

        data_dict = preprocess((sources, targets, logps), self.tokenizer, self.is_pretrain)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            # weights=weights,
            # logps=logps, 
            # ids=ids
        )
        
        if self.is_weighted_loss>=1 or model_args.round_id>=1:
            ret['weights'] = weights 
        
            ret['logps'] = logps # torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x[0]).to(device=labels.device) for x in logps], batch_first=True, padding_value=0.)
            ret['ids'] = ids 
        return ret 

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_data,eval_data = get_data(tokenizer=tokenizer, data_arg=(data_args.data_config, data_args.data_path),
                                      template_variation=data_args.template_variation)
    train_dataset,eval_dataset = SupervisedDataset(train_data),SupervisedDataset(eval_data)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, is_pretrain=data_args.is_pretrain==1, is_weighted_loss=data_args.is_weighted_loss>=1)
    if model_args.do_logp==1:
        return dict(
            # train_dataset=train_dataset, 
            
            eval_dataset=train_dataset, 
            data_collator=data_collator)
    else: 
        return dict(train_dataset=train_dataset, 
                    
                    # eval_dataset=eval_dataset, 
                    data_collator=data_collator)

def get_logp(shift_labels, loss_mask, shift_logits):
    shift_labels[loss_mask==0] = 0
    logprob = logprobs_from_logits(shift_logits, labels=shift_labels) # nsample, nseq, 
    masked_logps = []
    for mask,logp in zip(loss_mask, logprob):
        masked_logp = torch.masked_select(logp, mask==1).detach().cpu().numpy().astype(np.float32)
        masked_logps.append(masked_logp)
    return masked_logps
    
log_keys = ['pos_kl','neg_kl', 'pos_rho','neg_rho', 'pos_logp', 'neg_logp', 'pos_loss', 'neg_loss', 'syndata_logp', 'syndata_loss', 'syndata_rho']
class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        # Custom tensors to log, assuming you have them computed somewhere in your training loop
        # For example:
        
        super().on_log(args, state, control, **kwargs)
        for k in log_keys:
            custom_tensor_1 = kwargs.get(k, None)
            if custom_tensor_1 is not None:
                wandb.log({f"stats/{k}": custom_tensor_1})
            
            
class WeightedLossTrainer(Trainer):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_probs = dict()  # Initialize log_probs here
        self.log_keys = log_keys
        self.log_freq = 8*10 
        batchsize = 2 
        self.niter = 0
        self.log_queue = {k:RewardQueue(batchsize=self.log_freq*batchsize, report_num=self.log_freq*batchsize) for k in self.log_keys}

    def compute_loss(self, model, inputs, return_outputs=False):
        self.niter += 1
        weights = inputs.pop("weights", None)  # Extract and remove the weights from inputs
        token_reflogps = inputs.pop("logps", None)
        rho_version = training_args.use_rho
        has_reflogp = True
        if token_reflogps is None or model_args.do_logp==1: 
            has_reflogp = False 
        else: 
            sample_reflogps = [None if x is None else np.mean(x) for x in token_reflogps]
        # aa = token_reflogps[0]
        ids = inputs.pop("ids", None)
        num_sample = len(ids)
        outputs = model(**inputs)
        labels = inputs.get("labels")
        logits = outputs.get("logits")
        use_estimated_reward = data_args.rw_path != "None"
        if use_estimated_reward:
            rewards_ = [torch.tensor(weigh, dtype=torch.float32).to(logits.device) for weigh in weights]
            is_self = [len(x.shape)>0 for x in rewards_]
        else: 
            rewards_ = torch.tensor(weights, dtype=torch.float32).to(logits.device)
            is_self = [x is not None for x in token_reflogps]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = (shift_labels!=IGNORE_INDEX).float().to(device=shift_logits.device)
        
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        mask = loss_mask.view(num_sample, -1)
        
        tmp = self.loss_fct(shift_logits, shift_labels)
        
        tmp = tmp.view(num_sample, -1)
        
        if model_args.do_logp==1:
            for uid, mask,logp in zip(ids, loss_mask, -tmp):
                self.log_probs[uid] = torch.masked_select(logp, mask==1).detach().cpu().numpy()
        
        denom = mask.sum(-1)
        denom[denom==0.] = 1e-6
        token_loss = tmp * mask
        sample_loss = token_loss.sum(-1)/denom
        sample_logp = -sample_loss
        token_logp = -tmp 
        
        if has_reflogp:
            
            sample_estimated_kl = [0.0 if b is None else b-a for a,b in zip(sample_logp,sample_reflogps)]
            
            sample_rhos = torch.ones_like(token_logp)
            for i, (m, tlogp, isself) in enumerate(zip(mask, token_reflogps, is_self)):
                if isself:
                    tlogp = torch.from_numpy(tlogp.astype(np.float32)).to(token_logp.device)
                    a = token_logp[i][m==1]
                    b = tlogp[:len(a)] # 如果被truncate b lenth> a lenth
                    # print(b.shape, a.shape, (m==1).sum())
                    sample_rhos[i][m==1] = torch.exp(a-b).detach()
            sample_clipped_rhos = torch.clamp(sample_rhos, max=1.2)
            sample_rhos_display = (sample_clipped_rhos * mask).sum(-1)/(mask.sum(-1)+1e-6)
            
            # sample_clipped_rhos = torch.stack([torch.clamp(x, max=1.2) for x in sample_rhos])
            if len(sample_clipped_rhos.shape)==1:
                sample_clipped_rhos =  sample_clipped_rhos.view(num_sample, 1)
        token_rho_loss = token_loss * (sample_clipped_rhos if training_args.use_rho>0 and has_reflogp else 1.0)
        if use_estimated_reward:
            sample_rewards = torch.ones_like(token_logp) * 0.5 
            for i, (m, r, isself) in enumerate(zip(mask, rewards_, is_self)):
                if isself: 
                    sample_rewards[i][m==1] = (r-0.5)*2 
                
        else: sample_rewards = rewards_.view(num_sample, 1)
        
        token_rho_loss = token_rho_loss * sample_rewards
        sample_rho_loss = token_rho_loss.sum(-1) / denom
        loss = sample_rho_loss.mean()
        
        kl_penalty = 0.0*loss
        
        # log 
        kl_pos_weight = training_args.kl_weight
        kl_neg_weight = kl_pos_weight * training_args.kl_discount
        if has_reflogp:
            for r, v, lp, rh, b, ls in zip(rewards_, sample_estimated_kl, sample_logp, sample_rhos_display, token_reflogps, sample_rho_loss):
                if b is None: 
                    ver = 'syndata'
                else:
                    ver = 'neg' if r<0 else 'pos'
                
                if ver=='pos': kl_penalty += v * kl_pos_weight
                elif ver=='neg': kl_penalty += v * kl_neg_weight
                
                if b is not None:
                    name = f"{ver}_kl"
                    self.log_queue[name].append(v.item())
                
                name = f"{ver}_rho"
                self.log_queue[name].append(torch.mean(rh).item())  
                name = f"{ver}_logp"
                self.log_queue[name].append(lp.item())   
                name = f"{ver}_loss"
                self.log_queue[name].append(ls.item())   
                if ver=='neg':
                    print(ls, lp, r, rh)
            
        loss = loss + kl_penalty
        
        if self.accelerator.is_main_process and self.niter%self.log_freq==0:
            log_values = dict()
            for k in self.log_keys:
                tmp = self.log_queue[k].stats(running=True)[0]
                if tmp is None: continue 
                log_values[k] = tmp 
            self.callback_handler.on_log(self.args, self.state, self.control, logs=log_values)
            
        if return_outputs: 
            return (loss, outputs)
        else: return loss

    
def train():
    # wandb.login(relogin=True, key="")
    print_on_rank_0('Start Loading Model')
    print_on_rank_0(training_args)

    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            use_cache=True,
        ).to('cuda')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            #attn_implementation='eager'
        ).to('cuda')
    print_on_rank_0(model)
    print_on_rank_0('Start building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast = True,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side='right'
        # use_fast=False, # 二次训练有问题
    )
    # pdb.set_trace()
    print_on_rank_0("*"*50)
    print_on_rank_0("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print_on_rank_0("*"*50)
    print_on_rank_0("After adding, tokenizer length: ",len(tokenizer))

    print_on_rank_0('Start building data module')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    print_on_rank_0('Start building the trainer module')
    callbacks = []
    if model_args.do_logp==1:
        assert data_args.data_path.endswith('.pkl'), f"data_path {data_args.data_path} must be pickle"
        savefolder = data_args.data_path+'_logp'
        if dist.get_rank()==0:
            os.makedirs(savefolder, exist_ok=True)
        TrainerClass = WeightedLossTrainer
    elif  data_args.is_weighted_loss>0:
        TrainerClass = WeightedLossTrainer
        callbacks = [CustomWandbCallback()]
        print_on_rank_0('callbacks added:', callbacks)
    else:
        TrainerClass = Trainer
    
    print_on_rank_0('using trainer', str(TrainerClass))
    trainer = TrainerClass(model=model, 
                           tokenizer=tokenizer, 
                           args=training_args, 
                           callbacks=callbacks,
                           **data_module)

    # trainer.create_scheduler = lambda num_training_steps, optimizer: get_cosine_with_end_lr_scheduler(
    #     optimizer=optimizer,
    #     num_warmup_steps=training_args.num_warmup_steps,
    #     num_training_steps=num_training_steps, #training_args.num_cosine_steps,
    #     end_learning_rate=training_args.end_lr  # Set your desired end learning rate here
    # )
    if model_args.do_logp==1:
        eval_result = trainer.evaluate()
        # path = data_args.data_path
        savepath = savefolder+f'/{dist.get_rank()}.pkl'
        pkl.dump(trainer.log_probs, open(savepath,"wb"))
        print("writing", savepath, len(trainer.log_probs))
        
        exit(0)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # print_on_rank_0("pretrain")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

def get_cosine_with_end_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, end_learning_rate=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * progress))
        return cosine_decay * (1 - end_learning_rate) + end_learning_rate

    return get_scheduler(lr_lambda, optimizer, num_warmup_steps, num_training_steps)

if __name__ == "__main__":
    train()
