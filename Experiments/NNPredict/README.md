Use an existing 6x6x5 morphology.

Randomly generate 100 phaseoffset parameters.

Run 100 experiments and get the results.

# Execute

```bash
git pull
sh rebuild.sh -3f
python main.py
sbatch deepgreen.sh
```

result will be in `output.xml`.
