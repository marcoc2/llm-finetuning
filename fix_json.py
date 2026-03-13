import json
import re

fixed_lines = []
with open("dataset_spinoza.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            fixed_lines.append(line)
        except json.JSONDecodeError:
            # We must fix quotes inside the output or input field.
            # The structure is: {"instruction": "...", "input": "...", "output": "..."}
            idx_inst = line.find('"instruction": "') + len('"instruction": "')
            idx_in_key = line.find('", "input": "')
            
            inst = line[idx_inst:idx_in_key]
            
            idx_in_val = idx_in_key + len('", "input": "')
            idx_out_key = line.find('", "output": "')
            
            inp = line[idx_in_val:idx_out_key]
            
            idx_out_val = idx_out_key + len('", "output": "')
            out = line[idx_out_val:-2]  # until "}
            
            # Now let's escape any quotes inside inst, inp, out that are not escaped
            def escape_quotes(text):
                # temporarily replace \", then replace " with \", then restore
                text = text.replace('\\"', '\x00')
                text = text.replace('"', '\\"')
                text = text.replace('\x00', '\\"')
                return text

            inst = escape_quotes(inst)
            inp = escape_quotes(inp)
            out = escape_quotes(out)
            
            fixed_line = f'{{"instruction": "{inst}", "input": "{inp}", "output": "{out}"}}'
            try:
                json.loads(fixed_line)
                fixed_lines.append(fixed_line)
            except json.JSONDecodeError:
                print(f"Failed to fix line {i+1}")

with open("dataset_spinoza.jsonl", "w", encoding="utf-8") as f:
    for line in fixed_lines:
        f.write(line + "\n")
print(f"Fixed dataset, total valid lines: {len(fixed_lines)}")
