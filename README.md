# kontur_ai_detection
решение kaggle-соревнования от компании kontur

Для быстрого ознакомления посмотрите ноутбук с ключевыми моментами [[Solution_Pipeline]](./solution_pipeline.ipynb)  

Данный репозиторий содержит код, помогающий воспроизвести шаги моего решения

## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Download model weights
- Run `bash weights/download_weights.sh`


## (2) Quick start

### Run on a single image

This command runs the model on a single image, and outputs the uncalibrated prediction.

```
# Model weights need to be downloaded.
python demo.py -f examples/real.png -m weights/blur_jpg_prob0.5.pth
python demo.py -f examples/fake.png -m weights/blur_jpg_prob0.5.pth
```

### Run on a dataset

This command computes AP and accuracy on a dataset. See the [provided directory](examples/realfakedir) for an example. Put your real/fake images into the appropriate subfolders to test.

```
python demo_dir.py -d examples/realfakedir -m weights/blur_jpg_prob0.5.pth
```
