# Load model directly
import torch
# from prompt_utils import get_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_model import LlamaForCausalLM
import json
import argparse
# import utils
import pickle as pkl
from prompt_utils import *
from data_reader import BatchDatasetLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import pdb
import sys
import re

from eval_results import execute_with_timeout, eval_func
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '6066'
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_Pause_TOKEN = "<pause>"



parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--output", default='', type=str)
parser.add_argument("--stem_flan_type", default='', choices=['cot_prompt', 'cot_prompt2','pot_prompt', 'cot_trigger', 'pot_trigger', 'reflection','mix_trigger','ds_cot_trigger','ds_pot_trigger','qwen25_math_cot','qwen25_math_pot'], type=str)
parser.add_argument("--dtype", default='bfloat16', type=str)
parser.add_argument("--dataset", required=True, choices=[
    'gsm8k', 'svamp', 'math', 'numglue', 'gsm-hard', 'deepmind', 'simuleq','mawps', 'asdiv', 'mammoth_train'], type=str)
parser.add_argument("--data_path", type=str, default='')
parser.add_argument("--use_vllm", action='store_true', default=False)
parser.add_argument("--load_8bit", action='store_true', default=False)
parser.add_argument("--form", default='alpaca', type=str)
parser.add_argument("--shots", default=0, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--print", action='store_true', default=False)
parser.add_argument("--model_max_length", default=1024, type=int)
parser.add_argument("--cot_backup", action='store_true', default=False)
parser.add_argument("--prev_data_path", default="")
parser.add_argument("--use_logp", default=1, type=int)
parser.add_argument("--suffix", default="")
parser.add_argument(
    "--out_path",
    default=""
)
parser.add_argument("--num_total", default=0, type=int)
parser.add_argument("--deepseek_templ", default=0, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--sampling", default=0, type=int)

args = parser.parse_args()

nvida_code_start_token,nvidia_code_end_token = '<llm-code>', '</llm-code>'
code_start_token, code_end_token = "```python", "```"
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
sampling_numbers = 1

def _tokenize_fn(strings, tokenizer):
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
    return input_ids

tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            padding_side="right",
            truncation_side='right',
            model_max_length=2048,
            trust_remote_code=True,
            use_fast=True)
# special_tokens_dict = dict()
# if tokenizer.pad_token is None:
#     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
# if tokenizer.eos_token is None:
#     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
# if tokenizer.bos_token is None:
#     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
# if tokenizer.unk_token is None:
#     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
# tokenizer.add_special_tokens(special_tokens_dict)
def concat_tool_return(q,a,r):
    if '```python' not in a: return q
    na = a.split('```output')[0].strip() 
    final = f"{q}{na}\n```output\n{r}\n```\n"
    return final 

def run_question_only(questions: list, logp: bool):
    
    used_examples = get_examples(args.dataset, args.shots, args.stem_flan_type)
    # questions = questions[:10]
    # if args.use_vllm:
    prompt_no_input, prefix = get_prompt(used_examples, args.form)
    if args.shots == 8:
        prompt_no_input = ''
        for q,a in used_examples:
            prompt_no_input += "Problem:\n{query}\nSolution:\n{answer}\n".format(query=q,answer=a)
            prefix = "Problem:\n{query}\nSolution:"
        input_strs = [prompt_no_input+prefix.format(query=q) for q in questions]
    elif args.deepseek_templ==1:
        prompt_no_input = ''
        prefix = "User:{query}\n\nAssistant:"
        input_strs = [prefix.format(query=q) for q in questions]
    elif args.deepseek_templ==2: # do rerun 
        input_strs = [concat_tool_return(q,a,r) for q,a,r in questions]
    elif args.deepseek_templ==3: # qwen 2.5 math 
        input_strs = questions[:10]
    else: input_strs = [prompt_no_input+ prefix.format(query=q) for q in questions]
    # import pdb; pdb.set_trace()
    input_strs_ = input_strs 
    if args.deepseek_templ==2: # some inputs have certain results after round 1, we fix it 
        input_strs = []
        # forward_idx = []
        for idx, (q,a,r) in enumerate(questions):
            if '```python' not in a: continue
            input_strs.append(concat_tool_return(q,a,r)) 
            # forward_idx.append(idx)
    print('*'*10, 'final input')
    print(input_strs[0])
    
    response_outputs = llm.generate(input_strs, sampling_params)

    # if logp: output_logp = [[[id,probs[id].logprob] for id,probs in zip(output.outputs[0].token_ids,output.outputs[0].logprobs)] for output in response_outputs]
    # else: output_logp = None
    output_logp = None
    outputs = [output.outputs[0].text for output in response_outputs]
    #pdb.set_trace()
    outputs_ = outputs
    outputs = [x.split("Problem")[0] for x in outputs]
    
    matches_ = [None for _ in range(len(outputs))]
    if args.deepseek_templ==2: 
        outputs_  = []
        matches_ = []
        counter = 0
        for (q,a,r), inp in zip(questions, input_strs_):
            if '```python' not in a: 
                outputs_.append(a) # when solution not include program, directly save it
                matches_.append(r)
            else: 
                aa = inp.split('Assistant:')[-1]
                outputs_.append(aa+outputs[counter])
                matches_.append(None)
                counter += 1
    print('*'*10, 'final output')
    print(outputs_[0])
    
    return dict(solution=outputs_, logp=output_logp, req=input_strs_, match=matches_)
    

def run_question_loop(questions: list, logp: bool):
    
    used_examples = get_examples(args.dataset, args.shots, args.stem_flan_type)
    # questions = questions[:10]
    # if args.use_vllm:
    prompt_no_input, prefix = get_prompt(used_examples, args.form)
    prompt_no_input = ''
    prefix = "User:{query}\n\nAssistant:"
    input_strs = [prefix.format(query=q) for q in questions]
    # if args.shots == 8:
    #     raise Exception("fewshot not supported in multi-round function calling")
    #     # prompt_no_input = ''
    #     # for q,a in used_examples:
    #     #     prompt_no_input += "Problem:\n{query}\nSolution:\n{answer}\n".format(query=q,answer=a)
    #     #     prefix = "Problem:\n{query}\nSolution:"
    #     # input_strs = [prompt_no_input+prefix.format(query=q) for q in questions]
    # elif args.deepseek_templ==1:
        
    # # elif args.deepseek_templ==2: # do rerun 
    # #     input_strs = [concat_tool_return(q,a,r) for q,a,r in questions]
    # # elif args.deepseek_templ==3: # qwen 2.5 math 
    # #     input_strs = questions[:10]
    # else: input_strs = [prompt_no_input+ prefix.format(query=q) for q in questions]
    # import pdb; pdb.set_trace()
    results = {idx:dict(q=q,req=[rq]) for idx,(rq,q) in enumerate(zip(input_strs,questions))}
    # if args.deepseek_templ==2: # some inputs have certain results after round 1, we fix it 
    #     input_strs = []
    #     # forward_idx = []
    #     for idx, (q,a,r) in enumerate(questions):
    #         if '```python' not in a: continue
    #         input_strs.append(concat_tool_return(q,a,r)) 
    #         # forward_idx.append(idx)
    print('*'*10, 'final input')
    # print(input_strs[0])
    
    remaining = {k:d['req'][-1] for k,d in results.items()}
    nround = 0
    while True: 
        rqlist = list(remaining.items())
        response_outputs = llm.generate([xx[1] for xx in rqlist], sampling_params)
        outputs = {idx:output.outputs[0].text for (idx,_),output in zip(rqlist,response_outputs)}
        for idx,tt in outputs.items():
            if 'outs' not in results[idx]:
                results[idx]['outs'] = []
            
            hascode = '```python' in tt 
            results[idx]['outs'].append(tt)
            if hascode: # do execution, cannot end 
                code_match = re.search("```python(.*?)```", tt, re.DOTALL)
                
                if code_match:
                    string = code_match.group(1)
                    run_result = execute_with_timeout(string)
                    exe_result = run_result.get("result", None)
                    if exe_result is None: 
                        program_out = str(run_result.get("error", ""))
                    else: program_out = exe_result 
                else: program_out = "Code without closing marks."
                result = program_out
                if 'exe' not in results[idx]:
                    results[idx]['exe'] = []
                results[idx]['exe'].append(result)
                prev_req = results[idx]['req'][-1]
                sol = tt.split('```output')[0].strip()
                nrq = f"{prev_req}{sol}\n```output\n{result}\n```\n"
                results[idx]['req'].append(nrq)
                
            else: 
                prev_sol = results[idx]['req'][-1].split('Assistant:')[-1]
                sol = tt 
                final = f"{prev_sol}{sol}"
                
                results[idx]['solution'] = final
        nround += 1
        
        if nround>=5: break
        else:
            remaining = {k:d['req'][-1] for k,d in results.items() if 'solution' not in d}
    
    for k,d in results.items():
        if 'solution' not in d:
            prev_sol = d['req'][-1].split('Assistant:')[-1]
            sol = tt.split('```output')[0].strip()
            final = f"{prev_sol}{sol}"
            d['solution']  = final 

    matches_ = [None for _ in range(len(results))]
    
    num = len(results)
    solutions = [results[i]['solution'] for i in range(len(results))]
    
    reqs = [results[i]['req'][0] for i in range(num)]
    return dict(solution=solutions, logp=None, req=reqs, match=matches_)
    

if __name__ == "__main__":
    
    correct, wrong = 0, 0
    final_result_path = None
    
    def equal(a,b):
        flag = False
        try: 
            if abs(a-b)< 1e-7:
                flag = True
        except:
            flag = False
        return flag
    
    all_results = []
    if args.dataset.endswith("_train"):
        tmp = args.data_path.split(os.path.sep)[-1]
        real_dataset_name = 'train_' + tmp.split('.')[0]
        loader_name = args.dataset
        args.dataset = real_dataset_name
    else: 
        loader_name = args.dataset
        real_dataset_name = 'test_' + loader_name
    
    if args.deepseek_templ==2:
        loader_name = 'ds_inference'
    print(f"args.dataset={args.dataset}, loader_name={loader_name}")
    # assert args.output, "args.output must not be empty"
    # if not args.output:
    suffix = args.stem_flan_type.lower() #'PoT' if 'pot' in args.stem_flan_type.lower() else 'CoT'
    tmp = args.model.strip('/').split('/')
    for ii,kk in enumerate(tmp):
        if 'aa' in kk:
            break 
    ii += 1
    modelname = tmp[ii]
    modelname = modelname.replace('-', '_')
    
    folder = f"{args.out_path}/{args.suffix}{modelname}_s{args.sampling}_round{args.deepseek_templ}/{real_dataset_name}_{suffix}"
    os.makedirs(folder, exist_ok=True)
    args.output = f'{folder}/{args.num_total}_{args.rank}.json'
    
    print('Writing the output to', folder)
    # file_handle = open(args.output, 'w')
    tmp_path = args.output+'_tmp.pkl'
    ok_to_eval = False
    print(f"{args.out_path}/{args.suffix}{modelname}/{real_dataset_name}_{suffix}; {args.output}")
    if os.path.exists(tmp_path):
        try:
            pkl.load(open(tmp_path,"rb"))
            ok_to_eval = True 
        except:
            ok_to_eval = False
    if args.deepseek_templ==2:
        ok_to_eval = False
    tmp_path2 = f"{args.out_path}/{args.suffix}{modelname}_s{args.sampling}_round1/{real_dataset_name}_{suffix}/{args.num_total}_{args.rank}.json_matchlog.pkl"
    if args.deepseek_templ==2:
        args.data_path = tmp_path2
        assert os.path.exists(args.data_path), f"{args.data_path} must exist"
    if ok_to_eval:
        print("directly evaluating")
        eval_func(args.output)
        exit(0)
    
    # tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    if args.deepseek_templ==2: args.sampling = 0 # since we simply do multiturn based on first turn, do greedy
    if args.use_vllm:
        use_logp = args.sampling>0 and args.use_logp==1
        # "</s>", "<|im_end|>", "<|endoftext|>","User",'Assistant'
        stop_tokens = ["USER:", "USER", "ASSISTANT:", "ASSISTANT", "### Instruction:", "Response:", "Response", "<start_of_turn>", "[INST]","Problem","<|im_end|>","</s>", "<|im_end|>"]
        if args.sampling>0:
            sampling_params = SamplingParams(n=1,temperature=1, top_p=0.95, max_tokens=args.model_max_length, stop=stop_tokens, logprobs=1)
        else:
            sampling_params = SamplingParams(temperature=0,
                                         top_p=1,
                                         max_tokens=args.model_max_length,
                                         stop=stop_tokens)
        print("*"*10, 'sampling params')
        print(sampling_params)
        llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), dtype=args.dtype, trust_remote_code=True)

        args.batch_size = -1
        print('Using VLLM, we do not need to set batch size!')
        
        print('deactivating ray init')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_8bit=args.load_8bit,
            torch_dtype=DTYPES[args.dtype],
            trust_remote_code=True)
        
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.model,
        #     device_map="auto",
        #     load_in_8bit=args.load_8bit,
        #     torch_dtype=DTYPES[args.dtype],
        #     trust_remote_code=True)
        model.eval()
    # if args.deepseek_templ==2:
    #     args.batch_size = 10
    if args.prev_data_path!="":
        ref = pkl.load(open(args.prev_data_path,"rb"))
        q2acc = dict()
        for item in ref:
            q2acc[item['question'].strip()] = item['acc']
        do_continue = True
    else: do_continue = False
    print(loader_name, args.data_path)
    
    for questions, groundtruths in tqdm(BatchDatasetLoader(loader_name, args.batch_size, num_total=args.num_total, rank=args.rank, data_path=args.data_path, repeat=args.sampling)):
        # First pass to use PoT
        questions = questions
        reduced_qset = []
        if do_continue:
            for q in questions:
                if q.strip() not in q2acc: continue 
                if q2acc[q.strip()]>0: continue 
                reduced_qset.append(q)
        else: reduced_qset = questions
        print(f"after reduction: {len(questions)}>{len(reduced_qset)}")
        questions = reduced_qset
        
        if args.deepseek_templ==2:
            processed_questions = questions 
            questions = [x[0] for x in questions]
        else:
            processed_questions = process_question_with_flan_tag(questions, args.stem_flan_type)
            print('*'*10, 'single query')
            print(processed_questions[0])
            json.dump(processed_questions, open(args.output+'_tmp_q', "w"))
            print('dumped q')
       
        print('only prompt')
        
        if args.deepseek_templ>=3: run_question = run_question_loop
        else: run_question = run_question_only
        returned_dict = run_question(processed_questions, logp=use_logp)
        returned_dict.update(question=questions, correct=groundtruths)
        
        keys = ['question','solution','correct','req','match']
        if use_logp == 1:
            keys.append('logp')
        returned_values = [returned_dict[k] for k in keys]
        # we save it first 
        # import pdb; pdb.set_trace()
        inferred_results = []
        for tup in zip(*returned_values):
            example = {k:tup[idx] for idx,k in enumerate(keys)}
            example['task'] = args.dataset
            
            inferred_results.append(example)
        
        pkl.dump(inferred_results, open(args.output+'_tmp.pkl', "wb"))
        print('dumped inferred results')
        
    del llm
    eval_func(args.output)
    