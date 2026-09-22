import sys

path = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
with open(path, encoding="utf-8") as f:
    lines = f.read().splitlines()
for i, line in enumerate(lines, 1):
    if start <= i <= end:
        sys.stdout.write(f"{i}: {line}\n")