'''
Author: Robert Post

This class encompases the functionality to run a Deep Q Network agent as outlined in:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015
'''


import sys
import copy
import os
import cPickle
import time
import logging
import random
import numpy as np
import cv2
import argparse
import theano
import sys
import copy
import os
import cPickle
import time
import logging
import random
import numpy as np
import cv2
import argparse
import theano

sys.path.append("../utilities")
sys.path.append("../network")

import Parameters
import Environment
import DeepNetworks
import DeepQNetwork
import DQNAgentMemory


floatX = theano.config.floatX


class DQNAgent(object):
    def __init__(self, actionList, inputHeight, inputWidth, batchSize, phiLength, 
        nnFile, loadWeightsFlipped, updateFrequency, replayMemorySize, replayStartSize,
        networkType, updateRule, batchAccumulator, networkUpdateDelay,
        discountRate, learningRate, rmsRho, rmsEpsilon, momentum,
        epsilonStart, epsilonEnd, epsilonDecaySteps, evalEpsilon, useSARSAUpdate, kReturnLength):        
        self.actionList         = actionList
        self.numActions         = len(self.actionList)
        self.inputHeight        = inputHeight
        self.inputWidth         = inputWidth
        self.batchSize          = batchSize
        self.phiLength          = phiLength
        self.nnFile             = nnFile
        self.loadWeightsFlipped = loadWeightsFlipped
        self.updateFrequency    = updateFrequency
        self.replayMemorySize   = replayMemorySize
        self.replayStartSize    = replayStartSize
        self.networkType        = networkType
        self.updateRule         = updateRule
        self.batchAccumulator   = batchAccumulator
        self.networkUpdateDelay = networkUpdateDelay
        self.discountRate       = discountRate
        self.learningRate       = learningRate
        self.rmsRho             = rmsRho
        self.rmsEpsilon         = rmsEpsilon
        self.momentum           = momentum
        self.epsilonStart       = epsilonStart
        self.epsilonEnd         = epsilonEnd
        self.epsilonDecaySteps  = epsilonDecaySteps
        self.evalEpsilon        = evalEpsilon
        self.kReturnLength      = kReturnLength
        self.useSARSAUpdate     = useSARSAUpdate

        self.trainingMemory  =DQNAgentMemory.DQNAgentMemory((self.inputHeight, self.inputWidth), self.phiLength, self.replayMemorySize, self.discountRate)
        self.evaluationMemory=DQNAgentMemory.DQNAgentMemory((self.inputHeight, self.inputWidth), self.phiLength, self.phiLength * 2,    self.discountRate)

        self.episodeCounter  = 0 
        self.stepCounter     = 0
        self.batchCounter    = 0
        self.lossAverages    = []
        self.actionToTake    = 0

        self.epsilon = self.epsilonStart
        if self.epsilonDecaySteps != 0:
            self.epsilonRate = ((self.epsilonStart - self.epsilonEnd) / self.epsilonDecaySteps)
        else:
            self.epsilonRate = 0

        self.training = False

        self.network = DeepQNetwork.DeepQNetwork(self.batchSize, self.phiLength, self.inputHeight, self.inputWidth, self.numActions,
            self.discountRate, self.learningRate, self.rmsRho, self.rmsEpsilon, self.momentum, self.networkUpdateDelay,
            self.useSARSAUpdate, self.kReturnLength,
            self.networkType, self.updateRule, self.batchAccumulator)

        if self.nnFile is not None:
            #Load network
            DeepNetworks.loadNetworkParams(self.network.qValueNetwork, self.nnFile, self.loadWeightsFlipped)
            self.network.resetNextQValueNetwork()





    def agentCleanup(self):
        pass

    def startEpisode(self, observation):
        self.batchCounter= 0 
        self.lossAverages= []

        if self.training:
            self.epsilon = max(self.epsilonEnd, self.epsilonStart - self.stepCounter * self.epsilonRate)
        else:
            self.epsilon = self.evalEpsilon
        actionIndex      = np.random.randint(0, self.numActions - 1)
        returnAction     = self.actionList[actionIndex]
        self.actionToTake= actionIndex
        if self.training:
            self.trainingMemory.addFrame(observation)
        else:
            self.evaluationMemory.addFrame(observation)

        return returnAction

    def endEpisode(self, reward):
        self.episodeCounter += 1
        self.stepCounter    += 1

        if self.training:
            self.trainingMemory.addExperience(np.clip(reward, -1, 1), self.actionToTake, True)

        avgLoss = np.mean(self.lossAverages)
        return avgLoss
        


    '''
    This function receives the reward and next state and returns the action to take
    '''
    def stepEpisode(self, reward, observation):
        self.stepCounter += 1

        if self.training:
            currentMemory = self.trainingMemory
        else:
            currentMemory = self.evaluationMemory

        currentMemory.addExperience(np.clip(reward, -1, 1), self.actionToTake, False)
        currentMemory.addFrame(observation)

        if self.stepCounter >= self.phiLength:
            phi = currentMemory.getPhi()
            actionIndex = self.network.chooseAction(phi, self.epsilon)
        else:
            actionIndex = np.random.randint(0, self.numActions - 1)

        if self.training and len(self.trainingMemory) >= self.replayStartSize and self.stepCounter % self.updateFrequency == 0:
            loss = self.runTrainingBatch()
            self.batchCounter += 1
            self.lossAverages.append(loss)

        self.actionToTake = actionIndex
        return self.actionList[self.actionToTake]

    def runTrainingBatch(self):
        batchStates, batchActions, batchRewards, batchNextStates, batchNextActions, batchTerminals, batchTasks = self.trainingMemory.getRandomExperienceBatch(self.batchSize, kReturnLength = self.kReturnLength)
        return self.network.trainNetwork(batchStates, batchActions, batchRewards, batchNextStates, batchNextActions, batchTerminals)


    def startTrainingEpoch(self, epochNumber):
        self.training = True

    def endTrainingEpoch(self, epochNumber):
        pass
        
    def startEvaluationEpoch(self, epochNumber):
        self.training       = False
        self.episodeCounter = 0

    def endEvaluationEpoch(self, epochNumber):
        pass

    def computeHoldoutQValues(self, holdoutSize):
        holdoutStates = self.trainingMemory.getRandomExperienceBatch(holdoutSize)[0]
        holdoutSum = 0

        for i in xrange(holdoutSize):
            holdoutSum += np.mean(self.network.computeQValues(holdoutStates[i, ...]))

        return holdoutSum / holdoutSize


