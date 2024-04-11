git clone https://github.com/biubug6/Pytorch_Retinaface.git utils/Pytorch_Retinaface
mkdir data/generated-or-not/face_images
cp utils/detect.py utils/Pytorch_Retinaface/
mkdir utils/Pytorch_Retinaface/weights
python utils/Pytorch_Retinaface/detect.py
