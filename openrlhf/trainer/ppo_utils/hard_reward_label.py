
import re
import torch

from math_verify import verify, parse



def is_number(s):
    try:  
        float(s)
        return True
    except ValueError:  
        pass 
    try:
        import unicodedata  
        unicodedata.numeric(s)  
        return True
    except (TypeError, ValueError):
        pass
    return False


def reward_func(sequences, _, answers):
    format_r = []
    acc_r = []
    total_r = []


    for sequence, answer, __  in zip(sequences, answers, _):

        extracted = parse(sequence)
        if len(extracted) < 2:
            print(extracted)
        try:
            extracted[0] = int(extracted[0])
            extracted[1] = str(int(float(extracted[1])))
            accuracy = verify(extracted, answer)
            if accuracy:
                accuracy_reward = 1
            else:
                accuracy_reward = 0
        except (ValueError, TypeError, IndexError) as e:
            accuracy_reward = 0 
        
       
        boxed_pattern = r'boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, sequence)
        
        if not matches:
            format_reward = 0.0
        elif is_number(matches[-1]) and float(matches[-1]) == float(answer):
            format_reward = 1.0
        else:
            format_reward = 0.5



        
        total_reward = format_reward + accuracy_reward

        format_r.append(format_reward)
        acc_r.append(accuracy_reward)
        total_r.append(total_reward)



    return {
            "rewards": torch.tensor(total_r, dtype=torch.float32),
            "extra_logs": {
                "format_rewards": torch.tensor(format_r, dtype=torch.float32),
                "acc_rewards": torch.tensor(acc_r, dtype=torch.float32)
            }
        }

