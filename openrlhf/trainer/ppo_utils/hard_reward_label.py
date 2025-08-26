from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
import torch
extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
import re



def reward_func(sequences, _, answers):
    format_r = []
    acc_r = []
    total_r = []


    for sequence, answer, __  in zip(sequences, answers, _):

        extracted = parse(sequence, extraction_config=extraction_target)
        gold = answer
        try:
            pred = float(extracted[1])
            is_equal = abs(pred - float(gold)) < 1e-10
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            is_equal = False

        if is_equal:
            accuracy_reward = 1.0
        else:
            accuracy_reward = 0.0
        

        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, sequence)
        
        if not matches:
            format_reward = 0.0
        else:
            try:
                if float(matches[-1].strip()) == float(gold):
                    format_reward = 1.0
                else:
                    format_reward = 0.5
            except (ValueError, TypeError, AttributeError, IndexError) as e:
                format_reward = 0.5
        print(matches)
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

