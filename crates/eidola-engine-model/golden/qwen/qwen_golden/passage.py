"""The goldens' passage: prose and a little code, tokenised with the
checkpoint's own tokenizer and no special tokens."""

TEXT = '''The cartographer who mapped the river delta kept a ledger of every channel she had walked. Each entry gave the date, the depth at the deepest point she could reach with a weighted line, the width at high water, and a short note on the colour of the silt. The survey office asked only for the depths. She sent them the depths every quarter, folded into a brown envelope, and kept the rest of the ledger in a tin box under her bed.

After twelve years the delta had moved. Three of the channels on the official chart no longer carried water at low tide, and a fourth had opened where the chart showed a sandbar. The office learned this when a grain barge ran aground on a bank that was not supposed to exist. Its inspectors found that her quarterly depths, read in order, had predicted the shift years before, and that her notes on the silt explained why: the river had begun to carry clay from a quarry upstream, and the clay settled faster than sand.

A few lines of code make the same point about any series of readings. They keep a running mean and report the readings that stray from it by more than a chosen margin:

```python
def strays(readings, margin=0.5):
    total = 0.0
    flagged = []
    for count, value in enumerate(readings, start=1):
        total += value
        mean = total / count
        if abs(value - mean) > margin:
            flagged.append((count, value, round(mean, 2)))
    return flagged


depths = [3.1, 3.0, 3.2, 2.9, 2.4, 2.1, 1.8, 1.6]
for index, value, mean in strays(depths):
    print(f"reading {index}: {value:.1f} against a mean of {mean:.2f}")
```

Applied to the ledger, it flags nothing for the first eight years and then flags almost every reading, which is how the inspectors described the river to the board: steady for a long time, and then not.
'''
