import re, pathlib

src = pathlib.Path("run_all_tmp.sh").read_text().splitlines()
groups = {str(i): [] for i in range(4)}
current = []
current_gpu = None

for line in src:
    m = re.match(r"CUDA_VISIBLE_DEVICES=(\d)", line.strip())
    if m:
        # start of a new command block
        if current and current_gpu is not None:
            groups[current_gpu].extend(current + [""])
        current = [line]
        current_gpu = m.group(1)
    else:
        current.append(line)

# flush last block
if current and current_gpu is not None:
    groups[current_gpu].extend(current + [""])

for g, lines in groups.items():
    out = pathlib.Path(f"run_gpu{g}.sh")
    out.write_text("\n".join(["#!/usr/bin/env bash"] + lines).strip() + "\n")
    out.chmod(0o755)
    print(f"Wrote {out}")