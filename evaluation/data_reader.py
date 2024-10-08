from eval_results import delete_extra_zero
import json 
decoder = json.JSONDecoder()
def read_data(dataset):
    num_total = 1 
    rank = 0
    questions, answers = [], []
    if dataset == 'math':
        with open(
                'evaluation/MATH.json',
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
                'evaluation/gsm8k.jsonl'
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
    return questions, answers


class BatchDatasetLoader:

    def __init__(self, dataset: str, batch_size: int, num_total=0, rank=0, data_path='', repeat=0):
        inp, out = data_reader(dataset, num_total, rank, data_path=data_path)


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
    