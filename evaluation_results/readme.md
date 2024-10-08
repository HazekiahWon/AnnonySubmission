# About Evaluation Results
We release the inference results of the trained models. We train models on top of 7B models of LLama-3.1, Qwen2-Math, Deepseek-Math.

We include the inference results of the following models:
- SFT-ed models to ensure they can write code to solve math queries. These models are named as `Code4Math_SFT_<ModelFamily>`
- Ablative models used in the current paper, including: qwen and deepseek. These models are named as `Code4Math_<AblationType>_<ModelFamily>`.
- Models trained using the proposed method. These models are named as `AutoCode4Math_<ModelFamily>`.

Results on GSM8K and MATH are currently available. Each subfolder include: 
- a `matchfailure' pickle file shows the detail logs of why a response is considered unmatched with the gold answer
- a `matchlog' pickle file shows the matching info of each greedily sampled responses, including matching result, query, response, gold answer. 
- a `result' json file shows the meta info of the model performance, including accuracy and code-rate.



## ToDo
Inference results on the OOD sets used by Qwen