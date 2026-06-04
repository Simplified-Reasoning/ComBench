1. Parse the string payload into a Python object (e.g., using `ast.literal_eval`).
2. Verify the object is a list of lists representing a square $n \times n$ grid.
3. Check that $n = 9$ (the minimum feasible scale).
4. Check that every element in the grid is a string and is exactly one of "I", "M", or "O".
5. Iterate through each row and verify that the counts of "I", "M", and "O" are all exactly $n / 3$.
6. Iterate through each column and verify that the counts of "I", "M", and "O" are all exactly $n / 3$.
7. Extract all diagonals in both directions (top-left to bottom-right and top-right to bottom-left).
8. For each diagonal, if its length is a multiple of 3, verify that the counts of "I", "M", and "O" within that diagonal are all exactly `length / 3`.
