'''
 Author: Robert Post

 These functions run a standard DQN experiment accross multiple games or flavors of games for transfer learning experiments.
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
from ale_python_interface import ALEInterface
import theano

sys.path.append("utilities/")
sys.path.append("network/")
sys.path.append("agents/")

import Parameters
import Environment
import Preprocessing
import DeepNetworks
import DeepQTransferNetwork
import DQNAgentMemory
import DQTNAgent
import TransferTaskModule


def run_experiment(args):
    parameters = Parameters.processArguments(args, __doc__)

    #if the nnFile is a directory, check for a previous experiment run in it and start from there
    #load its parameters, append to its evalresults file, open its largest network file
    #If its none, create a experiment directory. create a results file, save parameters, save network files here. 

    experimentDirectory = parameters.rom + "_" + time.strftime("%d-%m-%Y-%H-%M") +"/"
    resultsFileName = experimentDirectory + "results.csv"
    startingEpoch = 0
    if parameters.nnFile is None or parameters.nnFile.endswith(".pkl"):
        #Create your experiment directory, results file, save parameters
        if not os.path.isdir(experimentDirectory):
            os.mkdir(experimentDirectory)

        resultsFile = open(resultsFileName, "a")
        resultsFile.write("Epoch,\tAverageReward,\tMean Q Value\n")
        resultsFile.close()

        parametersFile = open(experimentDirectory + "parameters.pkl" , 'wb', -1)
        cPickle.dump(parameters,parametersFile)
        parametersFile.close()


    if parameters.nnFile is not None and os.path.isdir(parameters.nnFile):
        #Found a experiment directory
        if not parameters.nnFile.endswith("/"):
            parameters.nnFile += "/"

        experimentDirectory = parameters.nnFile
        resultsFileName = experimentDirectory + "results.csv"

        if os.path.exists(experimentDirectory + "parameters.pkl"):
            parametersFile = open(experimentDirectory + "parameters.pkl" , 'rb')
            parameters = cPickle.load(parametersFile)
            parametersFile.close()
        else:
            parametersFile = open(experimentDirectory + "parameters.pkl" , 'wb', -1)
            cPickle.dump(parameters,parametersFile)
            parametersFile.close()

        contents = os.listdir(experimentDirectory)
        networkFiles = []
        for handle in contents:
            if handle.startswith("network") and handle.endswith(".pkl"):
                networkFiles.append(handle)

        if len(networkFiles) == 0:
            #Found a premature experiment, didnt finish a single training epoch
            parameters.nnFile = None
        else:
            #Found a previous experiments network files, now find the highest epoch number
            highestNNFile = networkFiles[0]
            highestNetworkEpochNumber = int(highestNNFile[highestNNFile.index("_") + 1 : highestNNFile.index(".")])
            for networkFile in networkFiles:
                networkEpochNumber =  int(networkFile[networkFile.index("_") + 1 : networkFile.index(".")])
                if networkEpochNumber > highestNetworkEpochNumber:
                    highestNNFile = networkFile
                    highestNetworkEpochNumber = networkEpochNumber

            startingEpoch = highestNetworkEpochNumber + 1
            #dont use full exploration, its not a good way to fill the replay memory when we already have a decent policy
            if startingEpoch > 4:
                parameters.epsilonStart = parameters.epsilonEnd

            parameters.nnFile = experimentDirectory + highestNNFile
            print "Loaded experiment: " + experimentDirectory + "\nLoaded network file:" + highestNNFile

    
    sys.setrecursionlimit(10000)
    ale = ALEInterface()

    Environment.initializeALEParameters(ale, parameters.seed, parameters.frameSkip, parameters.repeatActionProbability, parameters.displayScreen)




    # ale.loadROM(parameters.fullRomPath)

    # minimalActions = ale.getMinimalActionSet()

    # difficulties = ale.getAvailableDifficulties()
    # modes = ale.getAvailableModes()

    # maxNumFlavors = len(difficulties) * len(modes)

    # difficulties = createFlavorList(parameters.difficultyString, len(difficulties))
    # modes = createFlavorList(parameters.modeString, len(modes))

    # transferTaskModule = TransferTaskModule.TransferTaskModule(difficulties, modes)


    transferTaskModule = TransferTaskModule.TransferTaskModule(ale, parameters.roms, parameters.difficultyString, parameters.modeString)
    numActionsToUse = transferTaskModule.getNumTotalActions()
    print "Number of total tasks:" + str(transferTaskModule.getNumTasks()) + " across " + str(transferTaskModule.getNumGames()) + " games."
    print "Actions List:" + str(transferTaskModule.getTotalActionsList())
    # print "Num difficulties: " + str(len(difficulties)) + " num modes: " + str(len(modes)) + " numtasks: " + str(transferTaskModule.getNumTasks())
    # print "Modes: " + str(modes)
    # print "Difficulties: " + str(difficulties)

    numTransferTasks = transferTaskModule.getNumTasks()

    if (parameters.reduceEpochLengthByNumFlavors):
        parameters.stepsPerEpoch = int(parameters.stepsPerEpoch / numTransferTasks)

    agent = DQTNAgent.DQTNAgent(transferTaskModule.getTotalActionsList(), parameters.croppedHeight, parameters.croppedWidth, 
                parameters.batchSize, 
                parameters.phiLength,
                parameters.nnFile, 
                parameters.loadWeightsFlipped, 
                parameters.updateFrequency, 
                parameters.replayMemorySize, 
                parameters.replayStartSize,
                parameters.networkType, 
                parameters.updateRule, 
                parameters.batchAccumulator, 
                parameters.networkUpdateDelay,
                transferTaskModule,
                parameters.transferExperimentType,
                numTransferTasks,
                parameters.discountRate, 
                parameters.learningRate, 
                parameters.rmsRho, 
                parameters.rmsEpsilon, 
                parameters.momentum,
                parameters.epsilonStart, 
                parameters.epsilonEnd, 
                parameters.epsilonDecaySteps,
                parameters.evalEpsilon,
                parameters.useSARSAUpdate,
                parameters.kReturnLength)



    for epoch in xrange(startingEpoch, parameters.epochs + 1):
        agent.startTrainingEpoch(epoch)
        runTrainingEpoch(ale, agent, epoch, parameters.stepsPerEpoch, transferTaskModule)
        agent.endTrainingEpoch(epoch)

        networkFileName = experimentDirectory + "network_" + str(epoch) + ".pkl"
        DeepNetworks.saveNetworkParams(agent.network.qValueNetwork, networkFileName)

        if parameters.stepsPerTest > 0 and epoch % parameters.evaluationFrequency == 0:
            agent.startEvaluationEpoch(epoch)
            avgRewardPerTask = runEvaluationEpoch(ale, agent, epoch, parameters.stepsPerTest, transferTaskModule)
            holdoutQVals = agent.computeHoldoutQValues(3200)

            resultsFile = open(resultsFileName, 'a')
            resultsFile.write(str(epoch) + ",\t")
            resultsString = ""

            for avgReward in avgRewardPerTask:
                resultsString += str(round(avgReward, 4)) + ",\t"

            resultsFile.write(resultsString)
            resultsFile.write("\t" + str([round(x, 4) for x in holdoutQVals]) + "\n")
            resultsFile.close()

            agent.endEvaluationEpoch(epoch)

    agent.agentCleanup()

def runTrainingEpoch(ale, agent, epoch, stepsPerEpoch, transferTaskModule):
    stepsRemaining = stepsPerEpoch
    numEpisodes = 0
    print "Starting Training epoch: " + str(epoch)
    while stepsRemaining > 0:
        numEpisodes += 1

        lowestSamplesTask = agent.trainingMemory.getLowestSampledTask()
        transferTaskModule.changeToTask(lowestSamplesTask)
        # diff, mode = transferTaskModule.getTaskTuple(lowestSamplesTask)
        # ale.setMode(mode)
        # ale.setDifficulty(diff)

        startTime = time.time()
        stepsTaken, epsiodeReward, avgLoss = runEpisode(ale, agent, stepsRemaining, lowestSamplesTask)
        endTime = time.time() - startTime
        fps = stepsTaken / endTime
        stepsRemaining -= stepsTaken
        print "TRAINING: Task: "+ str(lowestSamplesTask) + " Steps Left: " + str(stepsRemaining) + "\tsteps taken: " + str(stepsTaken) + "\tfps: "+str(round(fps, 4)) + "\tepisode reward: " +str(epsiodeReward) + "\tavgLoss: " + str(avgLoss)        
        sys.stdout.flush()

def runEvaluationEpoch(ale, agent, epoch, stepsPerTest, transferTaskModule):
    print "Starting Evaluation epoch: " + str(epoch)
    taskAverageRewards = []
    for currentEpisodeTask in xrange(transferTaskModule.getNumTasks()):
        stepsRemaining = stepsPerTest
        numEpisodes = 0
        totalReward = 0
        
        transferTaskModule.changeToTask(currentEpisodeTask)
        # diff, mode = transferTaskModule.getTaskTuple(currentEpisodeTask)        
        # ale.setMode(mode)
        # ale.setDifficulty(diff)

        while stepsRemaining > 0:
            numEpisodes += 1

            startTime = time.time()
            stepsTaken, epsiodeReward, avgLoss = runEpisode(ale, agent, stepsRemaining, currentEpisodeTask)
            endTime = time.time() - startTime
            fps = stepsTaken / endTime
            stepsRemaining -= stepsTaken
            totalReward += epsiodeReward    
            print "EVALUATING: Task:" + str(currentEpisodeTask) + " Steps Left: " + str(stepsRemaining) + "\tsteps taken: " + str(stepsTaken) + "\tfps: "+str(round(fps, 4)) + "\tepisode reward: " +str(epsiodeReward) + "\tavgLoss: " + str(avgLoss)            
            sys.stdout.flush()
        taskAverageRewards.append(float(totalReward) / numEpisodes)

    return taskAverageRewards



def runEpisode(ale, agent, stepsRemaining, currentEpisodeTask):
    maxEpisodeDuration = 60 * 60 * 5 #Max game duration is 5 minutes, at 60 fps
    framesElapsed       = 0
    totalEpisodeReward  = 0
    ale_game_over       = False

    screenObservation = ale.getScreenRGB()

    preprocessedObservation = Preprocessing.preprocessALEObservation(screenObservation, agent.inputHeight, agent.inputWidth)
    action = agent.startEpisode(preprocessedObservation, currentEpisodeTask)

    while not ale_game_over and framesElapsed < stepsRemaining and framesElapsed < maxEpisodeDuration:

        framesElapsed += 1
        reward = ale.act(action)
        totalEpisodeReward += reward

        if ale.game_over():
          ale_game_over = True

        screenObservation = ale.getScreenRGB()
        preprocessedObservation = Preprocessing.preprocessALEObservation(screenObservation, agent.inputHeight, agent.inputWidth)

        action = agent.stepEpisode(reward, preprocessedObservation)

    ale.reset_game()
    avgLoss = agent.endEpisode(0)

    return framesElapsed, totalEpisodeReward, avgLoss


if __name__ == "__main__":
    run_experiment(sys.argv[1:])
