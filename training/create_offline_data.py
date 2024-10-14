import json
from glob import glob 
import os
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
import pickle as pkl
import sys 

from evaluation.prompt_utils import cot_trigger, pot_trigger, pot_prompt

begin = "Let's"
pot_better_prefix = f"{begin} write a program.\n"
code_templ = "```python\n{pot}\n```"

cot_better_prefix = f"{begin} reason step by step.\n"

use_args = len(sys.argv)>1
suffix = sys.argv[1] if use_args else "r1"
expname = sys.argv[2] if use_args else ""

tag = sys.argv[3] if use_args else ""
do_explore = 'explore' in tag
do_sample = 'small' in tag
pos_only = False # 'pos' in tag
isds = 'dsmath' in expname
req_templ = "User:{}\n\nAssistant:" if isds else "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
keep_trigger = 'keeptrigger' in tag # keep trigger now means, keep the trigger version, but for auto, we still do rjs

negbetter2 = 'negbetter2' in tag 
allow_all = 'all' in tag

auto_only = 'minimal' in tag # rft is better than rjs, I guess rjs is not bad, just that self-samples are better

max_num = 100 # negmax
target_success = 0.75
max_easy = 2
maxneg = 2

comment_pattern = re.compile(r'#(.*)$', re.MULTILINE)
responses = dict()
for k in ['auto', 'trigger']:
    try:
        fp = f"meta/{suffix}_{k}_q2responses_{expname}.pkl"
        q2responses = pkl.load(open(fp,"rb"))
    except:
        print(f'error loading {fp}')
        continue 
    responses[k] = q2responses

if len(responses)==0:
    raise Exception("no data found.")
qid = 0
new_data = []
qset = set()
q2id = dict()
uqset = set()
distributions = defaultdict(int)
qhascorrect = dict()

if len(responses)==1: 
    try:
        q2responses = responses['trigger']
    except:
        q2responses = responses['auto']
else: q2responses = responses['auto']


for q, info in tqdm(q2responses.items()):
    maxneg = 2
    
    if q not in q2id: 
        q2id[q] = len(q2id)
    qid = q2id[q]
    
    task = info['task']
    if 'aqu' in task: continue 
    qset.add(q)
    setname = task.split('_')[-1]
    correct = info['correct'][0]
    
    req = req_templ.format(q) # if isds else info['req']
    
    sources = {k:q2r[q] for k,q2r in responses.items()}
    key2acc = dict()
    
    for key in ['cot', 'pot']:
        nc = 0
        nt = 0 
        for sname,sinfo in sources.items():
            nc += sinfo[f"{key}_nc"]
            nt += sinfo[f"{key}_nt"]
        key2acc[key] = nc/(nt+1e-8)
    if key2acc['pot']>key2acc['cot']:
        use_key = 'pot'
    else:
        use_key = 'cot'
    
    if use_key=='cot' and key2acc['cot']==0: continue 

    
    if allow_all or keep_trigger or negbetter: keys = ['cot','pot']
    else: keys = [use_key]

    # this is only valid for coldstart scenario
    
    tmp = sources.get("trigger", None)
    if tmp is None: tmp = sources["auto"]
    alterlist = tmp['resps']['pot']
    alterlist2 = tmp['resps']['cot']
    alter_idxes = [idx for idx,entry in enumerate(alterlist) if entry['match']]
    alter_idxes2 = [idx for idx,entry in enumerate(alterlist2) if entry['match']]
    # sname == auto 
    for sname in ['auto','trigger']:
        sinfo = sources[sname]
        for key in keys: 
            acc = key2acc[key]
            rsplist = sinfo['resps'][key]
            key_acc = sinfo[f"{key}_nc"]/(sinfo[f"{key}_nt"]+1e-6)
            max_trigger = 1
            if keep_trigger and sname=='trigger': # trigger: too easy we don't train
                if key_acc>0.75:
                    if np.random.uniform()>0.5: continue 
                    else: max_trigger = 1 
            for idx,entry in enumerate(rsplist):
                text = entry['text']
                suffix2 = None
                req_ = sinfo['req'] # not info['req'] because it only refers to auto 
                use_trigger = key!=use_key 
                modified = False
                isneg = not entry['match']
                suffix2 = 'default'
                fqid = f"{qid}.{key}.{sname}.{idx}"
                if allow_all: 
                    pass 
                elif sname=='auto':
                    if not entry['match']:
                        if not negbetter: continue 
                        if np.random.uniform()>0.5:
                            if key=='pot' : 
                                contrastive_used = False
                                
                                if negbetter2 and not contrastive_used:
                                    suffix2 = 'potnegbetter'
                                
                            else:
                                suffix2 = 'cotnegbetter'
                                
                    # else:
                    if suffix2 =='default':
                        if use_trigger: continue 
                        else: 
                            suffix2 = 'better'
                    
                else: # trigger
                    if auto_only: continue 
                    
                    if not keep_trigger: continue 
                    if not entry['native']: continue # response and req not match
                    if max_trigger==0: continue 
                    suffix2 = 'trigger'
                    max_trigger-=1
                
                
                if suffix2.endswith('negbetter'):
                    if suffix2.startswith('cot'): 
                        idxes = alter_idxes 
                        alist = alterlist
                    else: 
                        idxes = alter_idxes2
                        alist = alterlist2
                    if len(idxes)==0: continue 
                    ii = np.random.choice(idxes)
                    if suffix2.startswith('cot'): 
                        new = alist[ii]['text'].split('a program.')[-1].strip()
                    else:
                        new = alist[ii]['text'].split('step by step.')[-1].strip()
                    if key == 'cot':
                        text = text.split('step by step.')[-1].strip()
                        tmp = text.split('\n')[:10]
                        text = '\n'.join(tmp)
                        text = f"{cot_better_prefix}{text}\nSorry, the above solution may be incorrect. It's better to write a program.\n{new}"
                    else:
                        text = text.split('a program.')[-1].strip()
                        text = f"{pot_better_prefix}{text}\nSorry, the above solution may be incorrect. It's better to simply reason step by step.\n{new}"
                    
                    # suffix2 = 'cotnegbetter'
                    modified = True
                    isneg = False
                    req_ = req # make sure request is made by raw q 
                
                elif suffix2=='contrastneg':
                    req_ = req # make sure request is made by raw q 
                    suffix2 = 'contrastneg'
                    isneg = True
                    
                elif suffix2=='trigger':
                    assert keep_trigger and sname=='trigger', f'wrong: {keep_trigger},{sname}'
                    if 'req' in entry: req_ = entry['req']
                    else:
                        if text.startswith('```'): 
                            trig = pot_prompt 
                        else: trig = cot_trigger 
                        req_ = req_templ.format(q+trig)
                    if 'reason step by step' not in req_ and 'a program' not in req_:
                        print(req_)
                        import pdb; pdb.set_trace()
                    
                    isneg = not entry['match']
                elif suffix2=='better':
                    isneg = not entry['match']
                    req_ = req # make sure request is made by raw q 
                else:
                    import pdb; pdb.set_trace()
                
                if modified: 
                    fqid += ".modified"
                    distributions['modified'] += 1
    
                newd = dict(qid=fqid,
                req=req_, 
                gold=correct,
                type=f"{key}.{sname}.{suffix2}",
                instruction=q,
                output=text,
                source=f"{'self' if sname=='auto' else 'explore'}.{task}.{key}",
                # nc=nnc,
                # nt=nnt,
                trigger_value=acc, 
                neg=isneg,
                logp=None)
                new_data.append(newd)
                distributions[f"{task}.{sname}"] += 1
                distributions[f"{task}.{entry['match']}"] += 1
                distributions[f"{sname}"] += 1
                distributions[f"{sname}.{entry['match']}"] += 1
                
assert len(new_data)==len({x['qid'] for x in new_data}), 'qid must be unique'
typeset = set()
for item in new_data:
    if item['type'] not in typeset:
        typeset.add(item['type'])
        print(f"type = {item['type']}")
        print(item['req'])
        print(item['output'])
        print('---------')
tmp = sorted(distributions.items(), key=lambda x: x[0])
for tup in tmp:
    print(tup)
print(f"in {len(qset)} queries {len(new_data)} samples, ({len(qhascorrect)} queries) has correct answers ")
if allow_all:
    print(f"make sure in <all> mode, trigger data are with triggered request: 0==", sum([float('reason step by step' not in item['req'] and 'a program' not in item['req']) for item in new_data if 'trigger' in item['type']]))
import pdb; pdb.set_trace()
outpath = f"data/offline_{tag}_{suffix}_{expname}.pkl"
print('written to ', outpath)
pkl.dump(new_data, open(outpath,"wb"))


