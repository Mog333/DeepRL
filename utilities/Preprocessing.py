import numpy as np
import cv2

def grayScaleALEObservation(observation):
    image = observation.view(np.uint8).reshape(observation.shape+(4,))[..., :3]
    image = image.reshape(210,160,3)
    greyscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return greyscaled

def resizeALEObservation(obsevation, resizedImageHeight, resizedImageWidth):
    return cv2.resize( obsevation, (resizedImageHeight, resizedImageWidth), interpolation=cv2.INTER_LINEAR)

def preprocessALEObservation(observation, resizedImageHeight, resizedImageWidth):  
    return cv2.resize( grayScaleALEObservation(observation), (resizedImageHeight, resizedImageWidth), interpolation=cv2.INTER_LINEAR)