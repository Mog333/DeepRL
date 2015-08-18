import numpy as np
import cv2

def preprocessALEObservation(observation, resizedImageHeight, resizedImageWidth):
        image = observation.view(np.uint8).reshape(observation.shape+(4,))[..., :3]
        image = image.reshape(210,160,3)
        greyscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return cv2.resize( greyscaled, (resizedImageHeight, resizedImageWidth), interpolation=cv2.INTER_LINEAR)