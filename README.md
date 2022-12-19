# IVR-CheckNoteContent

## Installation
```
conda create -n ivr python=3.8 -y
conda activate ivr
pip install -r requirements.txt

git clone https://github.com/cleanlab/cleanlab.git
```

## Get stratified data
Run the notebook: stratify_data.ipynb

## Preprocessing
```
python preprocessing.py
```

## Extract audio features (can be used for both training and testing data)
```
python extract_audio_feature.py
```

## Run CleanLab to clean data (using non-speech and audio features)
```
python cleanlab_no_transcript.py
```

## Find issues in testing data
```
python infer_by_day.py
```
