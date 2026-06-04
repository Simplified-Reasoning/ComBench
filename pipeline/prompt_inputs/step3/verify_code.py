import sys
import ast
from collections import Counter

def verify():
    payload = sys.stdin.read().strip()
    if not payload:
        print("Empty payload")
        return
        
    try:
        grid = ast.literal_eval(payload)
    except Exception as e:
        print(f"Parse error: {e}")
        return

    if not isinstance(grid, list):
        print("Grid must be a list of lists.")
        return

    n = len(grid)
    if n != 9:
        print(f"Expected n=9, got n={n}")
        return

    for i, row in enumerate(grid):
        if not isinstance(row, list):
            print(f"Row {i} is not a list.")
            return
        if len(row) != n:
            print(f"Row {i} has length {len(row)}, expected {n}.")
            return
        for j, cell in enumerate(row):
            if cell not in ("I", "M", "O"):
                print(f"Invalid cell at ({i}, {j}): {cell}")
                return

    # Check rows
    for i, row in enumerate(grid):
        counts = Counter(row)
        if counts["I"] != 3 or counts["M"] != 3 or counts["O"] != 3:
            print(f"Row {i} does not have exactly 3 of each letter.")
            return

    # Check columns
    for j in range(n):
        col = [grid[i][j] for i in range(n)]
        counts = Counter(col)
        if counts["I"] != 3 or counts["M"] != 3 or counts["O"] != 3:
            print(f"Column {j} does not have exactly 3 of each letter.")
            return

    # Check diagonals (Top-left to bottom-right: i - j = k)
    for k in range(-(n-1), n):
        diag = []
        for i in range(n):
            j = i - k
            if 0 <= j < n:
                diag.append(grid[i][j])
        if len(diag) % 3 == 0:
            counts = Counter(diag)
            req = len(diag) // 3
            if counts["I"] != req or counts["M"] != req or counts["O"] != req:
                print(f"Main-diagonal with offset {k} (length {len(diag)}) does not have exactly {req} of each letter.")
                return

    # Check anti-diagonals (Top-right to bottom-left: i + j = k)
    for k in range(2 * n - 1):
        diag = []
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                diag.append(grid[i][j])
        if len(diag) % 3 == 0:
            counts = Counter(diag)
            req = len(diag) // 3
            if counts["I"] != req or counts["M"] != req or counts["O"] != req:
                print(f"Anti-diagonal with sum {k} (length {len(diag)}) does not have exactly {req} of each letter.")
                return

    print("True")

if __name__ == "__main__":
    verify()
