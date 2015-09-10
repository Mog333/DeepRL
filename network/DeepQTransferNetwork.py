"""
Author: Robert Post
Based on code from Nathan Sprague
from: https://github.com/spragunr/deep_q_rl
"""

import DeepNetworks
import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle
import imp

class DeepQTransferNetwork(object):
    def __init__(self, batchSize, numFrames, inputHeight, inputWidth, numActions, 
        discountRate, learningRate, rho, rms_epsilon, momentum, networkUpdateDelay,
        transferExperimentType = "fullShare", numTransferTasks = 0, 
        networkType = "conv", updateRule = "deepmind_rmsprop", batchAccumulator = "sum", clipDelta = 1.0, inputScale = 255.0):
        
        self.batchSize          = batchSize
        self.numFrames          = numFrames
        self.inputWidth         = inputWidth
        self.inputHeight        = inputHeight
        self.inputScale         = inputScale
        self.numActions         = numActions
        self.discountRate       = discountRate
        self.learningRate       = learningRate
        self.rho                = rho
        self.rms_epsilon        = rms_epsilon
        self.momentum           = momentum
        self.networkUpdateDelay = networkUpdateDelay
        self.networkType        = networkType
        self.updateRule         = updateRule
        self.batchAccumulator   = batchAccumulator
        self.clipDelta          = clipDelta
        self.updateCounter      = 0
        self.numTransferTasks   = numTransferTasks
        self.transferExperimentType = transferExperimentType
        # self.useSharedTransferLayer = useSharedTransferLayer
        self.hiddenTransferLayer = None

        states     = T.tensor4("states")
        nextStates = T.tensor4("nextStates")
        rewards    = T.col("rewards")
        actions    = T.icol("actions")
        terminals  = T.icol("terminals")

        self.statesShared      = theano.shared(np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX))
        self.nextStatesShared  = theano.shared(np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX))
        self.rewardsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype=theano.config.floatX), broadcastable=(False, True))
        self.actionsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))
        self.terminalsShared   = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))

        self.qValueNetwork, self.hiddenTransferLayer  = DeepNetworks.buildDeepQTransferNetwork(
            self.batchSize, self.numFrames, self.inputHeight, self.inputWidth, self.numActions, self.transferExperimentType , self.numTransferTasks, convImplementation=self.networkType)

        qValues = lasagne.layers.get_output(self.qValueNetwork, states / self.inputScale)

        if self.networkUpdateDelay > 0:
            self.nextQValueNetwork, _ = DeepNetworks.buildDeepQTransferNetwork(
                self.batchSize, self.numFrames, self.inputHeight, self.inputWidth, self.numActions, self.transferExperimentType, self.numTransferTasks, convImplementation = self.networkType)
            self.resetNextQValueNetwork()
            nextQValues = lasagne.layers.get_output(self.nextQValueNetwork, nextStates / self.inputScale)

        else:
            nextQValues = lasagne.layers.get_output(self.qValueNetwork, nextStates / self.inputScale)
            nextQValues = theano.gradient.disconnected_grad(nextQValues)


        target = rewards + terminals * self.discountRate * T.max(nextQValues, axis = 1, keepdims = True)
        targetDifference = target - qValues[T.arange(self.batchSize), actions.reshape((-1,))].reshape((-1, 1))

        if self.clipDelta > 0:
            targetDifference = targetDifference.clip(-1.0 * self.clipDelta, self.clipDelta)

        if self.batchAccumulator == "sum":
            loss = T.sum(targetDifference ** 2)
        elif self.batchAccumulator == "mean":
            loss = T.mean(targetDifference ** 2)
        else:
            raise ValueError("Bad Network Accumulator. {sum, mean} expected")


        networkParameters = lasagne.layers.helper.get_all_params(self.qValueNetwork)

        if self.updateRule == "deepmind_rmsprop":
            updates = DeepNetworks.deepmind_rmsprop(loss, networkParameters, self.learningRate, self.rho, self.rms_epsilon)
        elif self.updateRule == "rmsprop":
            updates = lasagne.updates.rmsprop(loss, networkParameters, self.learningRate, self.rho, self.rms_epsilon)
        elif self.updateRule == "sgd":
            updates = lasagne.updates.sgd(loss, networkParameters, self.learningRate)
        else:
            raise ValueError("Bad update rule. {deepmind_rmsprop, rmsprop, sgd} expected")

        if self.momentum > 0:
            updates.lasagne.updates.apply_momentum(updates, None, self.momentum)

        lossGivens = {
            states: self.statesShared,
            nextStates: self.nextStatesShared,
            rewards:self.rewardsShared,
            actions: self.actionsShared,
            terminals: self.terminalsShared
        }

        self.__trainNetwork = theano.function([], [loss, qValues], updates=updates, givens=lossGivens)
        self.__computeQValues = theano.function([], qValues, givens={states: self.statesShared})


    def trainNetwork(self, stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch, tasksBatch):
        self.statesShared.set_value(stateBatch)
        self.nextStatesShared.set_value(nextStateBatch)
        self.actionsShared.set_value(actionBatch)
        self.rewardsShared.set_value(rewardBatch)
        self.terminalsShared.set_value(terminalBatch)

        if self.hiddenTransferLayer is not None:
            self.hiddenTransferLayer.setTaskIndices(tasksBatch)

        if self.networkUpdateDelay > 0 and self.updateCounter % self.networkUpdateDelay == 0:
            self.resetNextQValueNetwork()

        loss, qValues = self.__trainNetwork()
        self.updateCounter += 1
        return np.sqrt(loss)

    def computeQValues(self, state, currentTask):
        stateBatch = np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX)
        stateBatch[0, ...] = state
        self.statesShared.set_value(stateBatch)

        if self.hiddenTransferLayer is not None:
            currentTaskIndices = self.hiddenTransferLayer.getTaskIndices()
            currentTaskIndices[0, ...] = currentTask
            self.hiddenTransferLayer.setTaskIndices(currentTaskIndices)
            
        return self.__computeQValues()[0]


    def chooseAction(self, state, currentTask, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.numActions)

        qValues = self.computeQValues(state, currentTask)
        return np.argmax(qValues)

    def resetNextQValueNetwork(self):
        networkParameters = lasagne.layers.helper.get_all_param_values(self.qValueNetwork)
        lasagne.layers.helper.set_all_param_values(self.nextQValueNetwork, networkParameters)