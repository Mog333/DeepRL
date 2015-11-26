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
        discountRate, learningRate, rho, rms_epsilon, momentum, networkUpdateDelay, useSARSAUpdate, kReturnLength,
        transferExperimentType = "fullShare", numTransferTasks = 0, taskBatchFlag = 0,
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
        self.useSARSAUpdate     = useSARSAUpdate
        self.kReturnLength      = kReturnLength
        self.networkType        = networkType
        self.updateRule         = updateRule
        self.batchAccumulator   = batchAccumulator
        self.clipDelta          = clipDelta
        self.updateCounter      = 0
        self.taskBatchFlag      = taskBatchFlag
        self.numTransferTasks   = numTransferTasks
        self.transferExperimentType = transferExperimentType
        # self.useSharedTransferLayer = useSharedTransferLayer

        states     = T.tensor4("states")
        nextStates = T.tensor4("nextStates")
        rewards    = T.col("rewards")
        actions    = T.icol("actions")
        nextActions= T.icol("nextActions")
        terminals  = T.icol("terminals")

        self.statesShared      = theano.shared(np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX))
        self.nextStatesShared  = theano.shared(np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX))
        self.rewardsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype=theano.config.floatX), broadcastable=(False, True))
        self.actionsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))
        self.nextActionsShared = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))
        self.terminalsShared   = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))

        self.hiddenTransferLayers = []

        self.qValueNetwork, hiddenTransferLayer  = DeepNetworks.buildDeepQTransferNetwork(
            self.batchSize, self.numFrames, self.inputHeight, self.inputWidth, self.numActions, self.transferExperimentType , self.numTransferTasks, self.taskBatchFlag ,convImplementation=self.networkType)
        self.hiddenTransferLayers.append(hiddenTransferLayer)


        qValues = lasagne.layers.get_output(self.qValueNetwork, states / self.inputScale)

        if self.networkUpdateDelay > 0:
            self.nextQValueNetwork, nextHiddenTransferLayer = DeepNetworks.buildDeepQTransferNetwork(
                self.batchSize, self.numFrames, self.inputHeight, self.inputWidth, self.numActions, self.transferExperimentType, self.numTransferTasks, self.taskBatchFlag, convImplementation = self.networkType)
            self.hiddenTransferLayers.append(nextHiddenTransferLayer)

            self.resetNextQValueNetwork()
            nextQValues = lasagne.layers.get_output(self.nextQValueNetwork, nextStates / self.inputScale)
        else:
            nextQValues = lasagne.layers.get_output(self.qValueNetwork, nextStates / self.inputScale)
            nextQValues = theano.gradient.disconnected_grad(nextQValues)


        if self.useSARSAUpdate:
            target = rewards + terminals * (self.discountRate ** self.kReturnLength) * nextQValues[T.arange(self.batchSize), nextActions.reshape((-1,))].reshape((-1, 1))
        else:
            target = rewards + terminals * (self.discountRate ** self.kReturnLength) * T.max(nextQValues, axis = 1, keepdims = True)

        # target = rewards + terminals * self.discountRate * T.max(nextQValues, axis = 1, keepdims = True)
        targetDifference = target - qValues[T.arange(self.batchSize), actions.reshape((-1,))].reshape((-1, 1))

        # if self.clipDelta > 0:
            # targetDifference = targetDifference.clip(-1.0 * self.clipDelta, self.clipDelta)

        quadraticPart = T.minimum(abs(targetDifference), self.clipDelta)
        linearPart = abs(targetDifference) - quadraticPart

        if self.batchAccumulator == "sum":
            loss = T.sum(0.5 * quadraticPart ** 2 + self.clipDelta * linearPart)
            # loss = T.sum(targetDifference ** 2)
        elif self.batchAccumulator == "mean":
            loss = T.mean(0.5 * quadraticPart ** 2 + self.clipDelta * linearPart)
            # loss = T.mean(targetDifference ** 2)
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
            nextActions: self.nextActionsShared,
            terminals: self.terminalsShared
        }

        self.__trainNetwork = theano.function([], [loss, qValues], updates=updates, givens=lossGivens, on_unused_input='warn')
        self.__computeQValues = theano.function([], qValues, givens={states: self.statesShared})


    def trainNetwork(self, stateBatch, actionBatch, rewardBatch, nextStateBatch, nextActionBatch, terminalBatch, tasksBatch):
        self.statesShared.set_value(stateBatch)
        self.nextStatesShared.set_value(nextStateBatch)
        self.actionsShared.set_value(actionBatch)
        self.nextActionsShared.set_value(nextActionBatch)
        self.rewardsShared.set_value(rewardBatch)
        self.terminalsShared.set_value(terminalBatch)

        for transferLayer in self.hiddenTransferLayers:
            if transferLayer is not None:
                transferLayer.setTaskIndices(tasksBatch)

        if self.networkUpdateDelay > 0 and self.updateCounter % self.networkUpdateDelay == 0:
            self.resetNextQValueNetwork()

        loss, qValues = self.__trainNetwork()
        self.updateCounter += 1
        return np.sqrt(loss)

    def computeQValues(self, state, currentTask):
        stateBatch = np.zeros((self.batchSize, self.numFrames, self.inputHeight, self.inputWidth), dtype=theano.config.floatX)
        stateBatch[0, ...] = state
        self.statesShared.set_value(stateBatch)

        for transferLayer in self.hiddenTransferLayers:
            if transferLayer is not None:
                currentTaskIndices = transferLayer.getTaskIndices()
                currentTaskIndices[0, ...] = currentTask
                transferLayer.setTaskIndices(currentTaskIndices)
            
        return self.__computeQValues()[0]


    def chooseAction(self, state, currentTask, epsilon, actionsToSelectFrom = []):
        if actionsToSelectFrom == []:
            actionsToSelectFrom = [x for x in xrange(0, self.numActions)]

        if np.random.rand() < epsilon:
            index = np.random.randint(0, len(actionsToSelectFrom))
            return actionsToSelectFrom[index]

        qValues = self.computeQValues(state, currentTask)
        reducedQValues = [qValues[i] for i in xrange(0, self.numActions) if i in actionsToSelectFrom]

        return np.argmax(reducedQValues)

    def resetNextQValueNetwork(self):
        networkParameters = lasagne.layers.helper.get_all_param_values(self.qValueNetwork)
        lasagne.layers.helper.set_all_param_values(self.nextQValueNetwork, networkParameters)