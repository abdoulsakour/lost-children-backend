#!/bin/bash
mkdir -p models
curl -L https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat -o models/shape_predictor_68_face_landmarks.dat
curl -L https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat -o models/dlib_face_recognition_resnet_model_v1.dat