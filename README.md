Thank you for your interest in CLOVER. 

Find our paper draft [here](https://drive.google.com/file/d/1DwrNJHNhFNo_FgmKRECekY42H07K7JP6/view?usp=sharing])

We will be publishing our finalized code here at the end of March 2025. If you would like to view our pending repository, see [https://github.com/ut-amrl/object-identification](https://github.com/ut-amrl/object-identification).


## Usage
```bash
conda create -n clover python=3.12
conda activate clover
pip install -e .
# or
pip install -e .[dev]
```

```bash
python src/train.py experiment=<config_file_name>
```

## CODa Re-ID
```bash
pip install -e ./coda_reid

python coda_reid/get_global_3d_bbox.py
```
