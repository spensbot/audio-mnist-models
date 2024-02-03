from typing import List, Dict


# Written by ChatGPT
def dict_reduce(data: List[Dict[str, float]]) -> Dict[str, List[float]]:
    result = {}

    for entry in data:
        for key, value in entry.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    return result
