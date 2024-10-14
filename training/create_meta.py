import json
from glob import glob 
import os
from collections import defaultdict
from tqdm import tqdm
import re
import sys
import pickle as pkl
from evaluation import prompt_utils 

import numpy as np
cot_trigger = "Let's reason step by step."
pot_trigger = "Let's write a program."
comment_pattern = re.compile(r'#(.*)$', re.MULTILINE)
use_args = len(sys.argv)>1
prefix = sys.argv[1] if use_args else "r1"
# round = int(prefix[1]) 
# do_modify = round==0
expname = sys.argv[2] if use_args else "" #"dsbase_sft6_replet_htl_negalter_2e5_E1_pre0_w0_ds0"
version = sys.argv[3]
basefolder = "results"
if version == 'trigger':
    folders = [x for x in glob(basefolder+f'/{expname}/*') if 'cot_trigger' in x or 'pot' in x]
else: folders = [x for x in glob(basefolder+f'/{expname}/*') if 'cot_prompt' in x ]
assert len(folders)>0, "glob empty:"+basefolder+f'/{expname}'
# first repair match data
flag = True
alldata = dict()
for folder in folders:
    print('*'*10, folder)
    files = glob(folder+'/*.json_tmp.pkl')
    print(files)
    
    for file in files:

        tmp = file[:-8]
        tmplog = tmp+'_matchlog.pkl'
        # print(tmplog)
        ok_to_skip = False
        
        try:
            data = pkl.load(open(tmplog,'rb'))
            ok_to_skip = True
            alldata[tmplog] = data

        except:
            ok_to_skip = False 
        if ok_to_skip: 
            # print(file)
            continue 
        flag = False
        break 
        # print('deal with', file)
        # eval_func(tmp)
    if not flag: break 
        

if not flag:
    print('there are missing json_tmp, exit')
    exit(-1)

q2responses = dict()
q2set = defaultdict(set)
allq = set()
flag = False
for folder in folders:
    print('*'*10)
    
    for file, data in alldata.items():
        tmplist = file.split(os.path.sep)[-2].split('_')
        for idx, ee in enumerate(tmplist):
            if ee=='train':
                break
        taskname = tmplist[idx+1]
        
        for item in tqdm(data):
            q = item['question']
            tmp = q
            for trig in [cot_trigger, pot_trigger]:
                if trig in tmp:
                    tmp = tmp.split(trig)[-1].strip()
            
            raw_q = tmp
            allq.add(raw_q)
            
            text = item['solution']
            
            match = item['match'][0]['res']
            logp = item.get('logp', None)
            
            if '```python' in text: # todo: multi rounds of code?
                key = 'pot'
                
            elif 'print(' in text:
                key = 'pot'
            else: key = 'cot'
            if version == 'trigger': # response and request may not align after autocode
                if 'step by step' in item['req']:
                    trigger_type = 'cot'
                else: trigger_type = 'pot'
            else: trigger_type = key
            if match:
                if '```python' in text:
                    endindex = re.search('```python.*?```', text, re.DOTALL)
                    if endindex is None:
                        continue
                    # if do_modify: text = text[:endindex.end()]
                    
            
            if raw_q not in q2responses:
                q2responses[raw_q] = dict(
                    req=item['req'],
                    task=item.get('task', taskname), 
                    correct=item['correct'],
                    resps=dict(cot=[], pot=[])
                    )

            if text not in q2set[raw_q+key]:
                q2responses[raw_q]['resps'][key].append(dict(
                    match=match,
                    text=text,
                    logp=logp,
                    req=item['req'], # we note that since response may not match req, the saved global req may not match the context of the responses here.
                    native=trigger_type==key
                    ))
                q2set[raw_q+key].add(text)
                
               
for q,info in q2responses.items():
    for k,vlist in info['resps'].items():
        nc = sum([v['match'] for v in vlist])
        nt = len(vlist)
        info[f'{k}_nc'] = nc 
        info[f'{k}_nt'] = nt
         
debug = sum([len(info['resps']['cot'])>0 and len(info['resps']['pot'])>0 for q,info in q2responses.items()])
hascot = sum([len(info['resps']['cot'])>0 for q,info in q2responses.items()])
haspot = sum([len(info['resps']['pot'])>0 for q,info in q2responses.items()])
print(f"both cot and pot: {debug}, hacot: {hascot}, haspot: {haspot}, total q {len(q2responses)}")

savefile = f"meta/{prefix}_{version}_q2responses_{expname}.pkl"
print(f"written into {savefile}")
pkl.dump(q2responses, open(savefile,'wb'))