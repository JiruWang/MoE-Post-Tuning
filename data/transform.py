import datasets
from datasets import load_dataset
file = "train"
dataset = load_dataset("parquet", data_files=f"/root/OpenRLHF/data/{file}.parquet")
from copy import deepcopy
write_list = []
for d in dataset['train']:

    dict_str = {
        "input": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nLet $a$ and $b$ be the two real values of $x$ for which\\[\\sqrt[3]{x} + \\sqrt[3]{20 - x} = 2\\]The smaller of the two values can be expressed as $p - \\sqrt{q}$, where $p$ and $q$ are integers. Compute $p + q$.<|im_end|>\n<|im_start|>assistant",
        "answer": "118",
        "gt_answer": "118",
        "subject": "Intermediate Algebra",
        "level": 5,
        "question": "Let $a$ and $b$ be the two real values of $x$ for which\\[\\sqrt[3]{x} + \\sqrt[3]{20 - x} = 2\\]The smaller of the two values can be expressed as $p - \\sqrt{q}$, where $p$ and $q$ are integers. Compute $p + q$.",
        "ground_truth_answer": "118",
        "target": "118"
    }



    info = d['extra_info']
    dict_str['input'] = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"+ info['question'] + "<|im_end|>\n<|im_start|>assistant"
    dict_str['answer'] = info["answer"]
    dict_str['gt_answer'] = d['reward_model']["ground_truth"]
    dict_str['subject'] = ""
    dict_str['level'] = ""
    dict_str['question'] = info["question"]
    dict_str['ground_truth_answer'] = d['reward_model']["ground_truth"]
    dict_str['target'] = d['reward_model']["ground_truth"]
    write_list.append(deepcopy(dict_str))
import json
with open(f"{file}.json", 'w', encoding='utf-8') as f:
    json.dump(write_list, f, ensure_ascii=False, indent=4)