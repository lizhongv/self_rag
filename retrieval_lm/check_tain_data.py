import json
with open("full_output_1005.jsonl", 'r') as f:
    lines = f.readlines()

examples = [json.loads(l) for l in lines]

print("finish.")