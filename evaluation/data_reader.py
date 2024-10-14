from eval_results import delete_extra_zero
import json 
decoder = json.JSONDecoder()
def read_data(dataset, data_path):
    num_total = 1 
    rank = 0
    questions, answers = [], []
    if dataset == 'math':
        with open(
                'MATH.json',
                'r') as f:
            loaded = json.load(f)
        if num_total > 0:
            num_each = len(loaded) // num_total + 1
            loaded = loaded[num_each * rank:num_each * (rank + 1)]
        for d in loaded:
            questions.append(d['question'])
            answers.append(d['answer'])

    elif dataset == "gsm8k":
        with open(
                'gsm8k.jsonl'
        ) as f:
            lines = f.readlines()
            if num_total > 0:
                num_each = len(lines) // num_total + 1
                lines = lines[num_each * rank:num_each * (rank + 1)]
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                for x in range(1):
                    questions.append(json_res["question"].strip())
                    answers.append(delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", "")))
    elif dataset.endswith('train'):
        print(f"datapath={data_path}")
        if data_path.endswith('json'):
            data = json.load(open(data_path))
            if num_total > 0:
                num_each = len(data) // num_total + 1
                data = data[num_each * rank:num_each * (rank + 1)]
            print(f"final data length = {len(data)}")
            for d in data:
                questions.append(d['question'])
                ans_str = d['gold_list'][0]
                ans_float = None
                try:
                    ans_float = eval(ans_str)
                except:
                    pass
                if ans_float is not None and not isinstance(
                        ans_float, float) and not isinstance(ans_float, int):
                    # print(ans_str, ans_float is None,
                    #       isinstance(ans_float, float),
                    #       isinstance(ans_float, int))
                    # ans_str = extract_math_answer(ans_str)
                    ans_float = None

                answers.append([ans_str, ans_float])
        else:
            with open(
                    data_path
            ) as f:
                lines = f.readlines()
                if num_total > 0:
                    num_each = len(lines) // num_total + 1
                    lines = lines[num_each * rank:num_each * (rank + 1)]
                for line in lines:
                    
                    d = json.loads(line)
                    if 'output' in d:
                        questions.append([d['question'],d['output']])
                    else:
                        questions.append(d['question'])
                    ans_str = d['gold_list'][0]
                    ans_float = None
                    try:
                        ans_float = eval(ans_str)
                    except:
                        pass
                    if ans_float is not None and not isinstance(
                            ans_float, float) and not isinstance(ans_float, int):
                        # print(ans_str, ans_float is None,
                        #       isinstance(ans_float, float),
                        #       isinstance(ans_float, int))
                        # ans_str = extract_math_answer(ans_str)
                        ans_float = None
                    
                    answers.append([ans_str, ans_float])
    else: raise Expection(f"{dataset} not supported for now.")
    return questions, answers


class BatchDatasetLoader:

    def __init__(self, dataset: str, batch_size: int, num_total=0, rank=0, data_path='', repeat=0):
        inp, out = read_data(dataset, data_path)


        self.inputs, self.outputs = [],[]
        if repeat==0: num_repeat = 1 
        else: 
            num_repeat = repeat
        for _ in range(num_repeat):
            self.inputs.extend(inp)
            self.outputs.extend(out)
        
        print(f"after repeating, num = {len(self.inputs)}")
        self.index = 0
        self.batch_size = batch_size
        self.length = len(self.inputs)
        print(self.length, self.batch_size)

    def __len__(self):
        if self.batch_size == -1:
            return 1
        else:
            return self.length // self.batch_size

    def __getitem__(self, index):
        if self.batch_size == -1:
            if index >= self.__len__():
                raise StopIteration
            else:
                return self.inputs, self.outputs
        else:
            if self.length % self.batch_size == 0:
                if index >= self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs = [], []
                    for i in range(
                            index * self.batch_size,
                            min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                    return tmp_inputs, tmp_outputs
            else:
                if index > self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs = [], []
                    for i in range(
                            index * self.batch_size,
                            min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                    return tmp_inputs, tmp_outputs
    