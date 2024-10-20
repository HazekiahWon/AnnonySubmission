We collect 119274 **unique** queries from publicly available math instruction-tuning datasets: (a) MetaMath (90823 queries), (b) Mammoth (24246 queries), (c) MMOS (2720 queries), (d) Openmath (1485 queries).

Each entry in `data.json` has the following key-values:
- q: the question
- a: the reference solution
- s: the source dataset
- gold: the extracted golden answer

We use the extracted golden answer and the evaluation scripts in `evaluation` to compute the outcome of the model inference results over training queries. We note that the extracted golden answer can be erroneous, as we simply perform rule-based extraction.