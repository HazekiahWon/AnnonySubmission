# Evaluation Scripts

Currently uploaded the deepseek scripts for evaluating GSM8k and MATH. Example of usage:
`bash deepseek_eval.sh {numshot} {promptver} {modelpath} 0 {deepseek_option} {datasetname} {outpath}`. 

Here:

- `numshot=0` for zero-shot, 
- `datasetname` now supports `{'gsm8k','math'}'`, 
- `promptver` now supports: `cot_prompt` for no explicit dictation (i.e., AutoCode), `cot_trigger` for explicitly instructing CoT, `pot_trigger` for explicitly instructing code. 
- `deepseek_option=3` for multi-rounds of code generation in deepseek, `deepseek_option=0` for default one round of code generation for other models 
- `outpath` is the output folder where the inference results will place
- `modelpath` is the model path.
