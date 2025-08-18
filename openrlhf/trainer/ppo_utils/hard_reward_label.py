import re 
from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal

import torch



def check_boxed(sequence):
    if re.search(r'\\boxed\{[^}]+\}', sequence):
        return True
    else:
        return False


def reward_func(sequences, _, answers):
    box_match_l = []

    for sequence, answer, __  in zip(sequences, answers, _):
        extract_answer = qwen_extract_answer(sequence.replace("<|endoftext|>", ""), data_name="math") #TODO: check the data_name, hard code here for now
        if qwen_math_equal(prediction=extract_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = -0.5
            
        if "boxed" not in sequence:
            box_match = -1.0
        print("*******************")
        print(box_match)
        box_match_l.append(box_match)
    return {
        "rewards": torch.tensor(box_match_l, dtype=torch.float32),}

