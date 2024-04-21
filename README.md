# kontur_ai_detection
решение kaggle-соревнования от компании kontur

Для быстрого ознакомления посмотрите ноутбук с ключевыми моментами [[Solution_Pipeline]](./solution_pipeline.ipynb)  

Данный репозиторий содержит код, помогающий воспроизвести шаги моего решения

## (1) Setup

### Зависимости, гарантирующие, что все получится!
- `pip install -r requirements.txt`

### Загрузите веса в /weights, если хотите инференс [[Model Weights]](https://drive.google.com/drive/folders/1HrWvw5s-9Ejj5KAMXaPgbQLIoNRQGred?usp=share_link) 


## (2) Быстрый Инференс

### На одной фотографии

```
# Model weights need to be downloaded.
python inference.py --use_ensemble visualization/kontur.jpg
python inference.py --use_ensemble visualization/yandex_vae.png
```
**Prob of being fake is 0.001**         
                                                        
<img src="visualization/kontur.jpg" alt="Image" height="450" width="500">

**Prob of being fake is 0.345**

<img src="visualization/yandex_vae.png" alt="Image" height="450" width="500">
