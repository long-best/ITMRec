## ITMRec

### Overview

### Environment

pip install -r requirements.txt

### Data

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1M4HH4EXHgGBD0kK9qnZIDEI1idbzM4nd?usp=drive_link).
The data contains text and image features extracted from Sentence-Transformers and VGG-16 and has been publiced in [MMRec](https://github.com/enoche/MMRec) framework.

### Run
1. Put your downloaded data (e.g. baby) under `data/` dir.
2. Run `main.py` to train ITMRec:
  `python main.py`
You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`. 

### Citation
