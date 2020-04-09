# LogoClassification
This project shows how to train and predict 3 classes of logo with keras and tensorflow lite

## Dependencies
python3, keras 2.2.4, tensorflow 1.15.0

## Train your images
Replace the current image folders in dataset/raw-img/train and dataset/raw-img/validation by your image folders.<br/>
Then <br/>
```bash
python3 train.py
```
## Convert your model to tflite
```bash
python3 convert2tflite.py
```

## Inference with .h5
```bash
python3 predict.py
```

## Inference with .tflite
```bash
python3 inference_on_tflite.py
```
