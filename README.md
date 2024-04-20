# kontur_ai_detection
решение kaggle-соревнования от компании kontur

Для быстрого ознакомления посмотрите ноутбук с ключевыми моментами [[Solution_Pipeline]](./solution_pipeline.ipynb)  

Данный репозиторий содержит код, помогающий воспроизвести шаги моего решения

## (1) Setup

### Зависимости, гарантирующие, что все получится!
- `pip install -r requirements.txt`

### Загрузите веса, если хотите инференс [[Model Weights]](https://drive.google.com/drive/folders/1HrWvw5s-9Ejj5KAMXaPgbQLIoNRQGred?usp=share_link) 

- `bash weights/download_weights.sh`


## (2) Быстрый Инференс

### На одной фотографии

This command runs the model on a single image, and outputs the uncalibrated prediction.

```
# Model weights need to be downloaded.
python inference.py -f examples/real.png -m weights/blur_jpg_prob0.5.pth
python inference.py -f examples/fake.png -m weights/blur_jpg_prob0.5.pth
```

### На датасете

This command computes AP and accuracy on a dataset. See the [provided directory](examples/realfakedir) for an example. Put your real/fake images into the appropriate subfolders to test.

```
python demo_dir.py -d examples/realfakedir -m weights/blur_jpg_prob0.5.pth
```
