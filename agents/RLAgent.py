###
# Author: Robert Post
###

'''
This is a simple base class for a Reinforcement Learning Agent
It acts randomly
'''
import numpy as np

class RLAgent(object):
    def __init__(self, actionList):
        self.actionList = actionList
        self.numActions = len(actionList)

    def agentCleanup(self):
        pass

    def startEpisode(self, observation):
        pass

    def endEpisode(self, reward):
        pass

    '''
    This function receives the reward and next state and returns the action to take
    '''
    def stepEpisode(self, reward, observation):
        actionIndex = np.random.randint(self.numActions)
        self.lastActionIndexTaken = actionIndex
        return self.actionList[actionIndex]

    def startLearningEpoch(self, epochNumber):
        pass
    def endLearningEpoch(self, epochNumber):
        pass
    def startEvaluationEpoch(self, epochNumber):
        pass
    def endEvaluationEpoch(self, epochNumber):
        pass
