# coffee2docker
build coffee2 docker image

- step1 Preprocessing Data
# resize and transfor image to grey scale
python3 dataset/preprocessData.py

- step2 Create tfrecord file
# create tfrecord file
python3 dataset/tfrecordCreate.py

- step3 Run the model
# run model and save weights
cd model/
python3 buildmodel.py
