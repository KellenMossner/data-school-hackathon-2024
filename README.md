# Patch Perfect: Using AI to Fix Roads
Stellenbosch University

### Team: Chi-Squared and Confused:
| Name           | SU Number |
|----------------|-----------|
| David Nicolay  | 26296918  |
| Jonty Donald   | 25957848  |
| Justin Dietrich| 25924958  |
| James Milne    | 25917307  |
| Kellen Mossner | 26024284  |S

## Execution
1. Run the Computer Vision model to segment images.
```bash
python3 src/segment.py
```

2. Run the script to remove duplicate segmentation outputs.
```bash
python3 src/rem_dup.py
```

3. Run the script to generate the final output.
```bash
python3 src/predict.py
```