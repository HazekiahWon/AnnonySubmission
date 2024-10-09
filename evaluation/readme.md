# Evaluation Scripts

Currently uploaded the deepseek scripts for evaluating GSM8k and MATH. 
The script first perform inference over the given testset `datasetname` with the specified model `modelpath`, and then do evaluation and save the results in `outpath`.

Example of usage:
`bash deepseek_eval.sh {numshot} {promptver} {modelpath} 0 {round_option} {datasetname} {outpath}`. 

Here:

- `numshot=0` for zero-shot, 
- `datasetname` now supports `{'gsm8k','math'}'`, 
- `promptver` now supports: `cot_prompt` for no explicit dictation (i.e., AutoCode), `cot_trigger` for explicitly instructing CoT, `pot_trigger` for explicitly instructing code. 
- `round_option=3` for multi-rounds of code generation in deepseek, `deepseek_option=0` for default one round of code generation for other models 
- `outpath` is the output folder where the inference results will place
- `modelpath` is the model path.

Note: the bash command should be run under the current folder. 
