# Mediapipe examples

From:
https://google.github.io/mediapipe/


## Installation

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For iris example, download `iris_landmark.tflite` into `models` directory:
```
wget -P models https://github.com/google/mediapipe/raw/master/mediapipe/modules/iris_landmark/iris_landmark.tflite
```

## RUN

One of the following:

```
python face_detection.py
python face_mesh.py
python hands.py
python head_posture.py
python holistic.py
python iris.py
python objectron.py
python pose.py
python selfie_segmentation.py
```

## Numpy

See example `python pose.py` for how to extract numpy array from the mediapipe landmark objects.
