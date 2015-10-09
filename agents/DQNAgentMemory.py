"""
Author: Robert Post
UOfAlberta 2015

This Class stores samples of experience - Only the most recent x samples are stored.
Samples are stores in a circular array.
Memory is used to construct random batches for the DQN.
"""

import time
import numpy as np
import theano

floatX = theano.config.floatX

class DQNAgentMemory(object):

    def __init__(self, stateShape, phiLength=4, memorySize=10000, kReturnLength = 1, numTasks = 1):
        """ 
        Arguments:
            stateShape - tuple containing the dimensions of the experiences being stored
            phiLength - number of images in a state.
            memorySize - The number of experiences that can be stored

            An experience is a single frame, a state is several frames (phiLength) in a single numpy tensor
        """

        self.currentMemoryIndex     = 0
        self.numberOfExperiences    = 0
        self.memorySize             = memorySize
        self.stateShape             = stateShape
        self.phiLength              = phiLength
        self.numTasks               = numTasks
        self.kReturnLength          = kReturnLength
        self.taskSampleCount        = np.zeros(self.numTasks, dtype='int32')

        self.stateMemory            = np.zeros((self.memorySize,) + self.stateShape , dtype = 'uint8')
        self.rewardMemory           = np.zeros(self.memorySize, dtype = floatX)
        self.actionMemory           = np.zeros(self.memorySize, dtype='int32')
        self.terminalMemory         = np.zeros(self.memorySize, dtype='int32')
        self.taskMemory             = -1 * np.ones(self.memorySize, dtype='int32')




    def addFrame(self, frame, memoryIndex = None):
        assert( memoryIndex == None or ( (memoryIndex < memorySize) and (memoryIndex >= 0) ) )
        if memoryIndex == None:
            memoryIndex = self.currentMemoryIndex

        assert(self.stateShape[0] == frame.shape[0])
        assert(self.stateShape[1] == frame.shape[1])

        self.stateMemory[memoryIndex, ...] = frame

    def addExperience(self, reward, action, terminal = 0, taskIndex = 0, memoryIndex = None):
        assert( memoryIndex == None or ( (memoryIndex < memorySize) and (memoryIndex >= 0) ) )
        if memoryIndex == None:
            memoryIndex = self.currentMemoryIndex

        self.actionMemory[memoryIndex]     = action
        self.rewardMemory[memoryIndex]     = reward
        self.terminalMemory[memoryIndex]   = terminal

        if self.taskMemory[memoryIndex] != -1:
            #Overwritting another memory
            self.taskSampleCount[self.taskMemory[memoryIndex]] -= 1
        self.taskSampleCount[taskIndex] += 1
        self.taskMemory[memoryIndex]       = taskIndex

        self.currentMemoryIndex = (self.currentMemoryIndex  + 1) % self.memorySize
        self.numberOfExperiences += 1


    def getPhiIndices(self, index = None):
        assert index < self.memorySize
        assert index < self.numberOfExperiences

        if index == None:
          index = self.currentMemoryIndex

        startingIndex = (index - self.phiLength + 1) % self.memorySize
        phiIndices = [(startingIndex + i) % self.memorySize for i in xrange(self.phiLength)]
        return phiIndices

    def getPhi(self, index = None):
        phiIndices = self.getPhiIndices(index)
        phi = np.array([self.stateMemory[i] for i in phiIndices])
        return phi

    def getRandomExperienceBatch(self, batchSize, taskIndex = None):
        assert batchSize < self.numberOfExperiences - self.phiLength + 1

        experienceStateShape = (batchSize, self.phiLength) + self.stateShape

        batchStates     = np.empty(experienceStateShape, dtype=floatX)
        batchNextStates = np.empty(experienceStateShape, dtype=floatX)
        batchRewards    = np.empty((batchSize, 1),       dtype=floatX)
        batchActions    = np.empty((batchSize, 1),       dtype='int32')
        batchNextActions= np.empty((batchSize, 1),       dtype='int32')
        batchTerminals  = np.empty((batchSize, 1),       dtype='int32')
        batchTasks      = np.empty(batchSize,            dtype='int32')

        count = 0
        maxIndex = min(self.numberOfExperiences, self.memorySize)

        while count < batchSize:
          index = np.random.randint(0, maxIndex - 1)
          if (index + 1 >= self.currentMemoryIndex) and (index < (self.currentMemoryIndex + self.phiLength)):
            continue

          phiIndices = self.getPhiIndices(index)

          if True in [self.terminalMemory[i] for i in phiIndices]:
            continue

          if taskIndex != None and self.taskMemory[i] != taskIndex:
            continue

          batchStates[count]     = self.getPhi(index)
          batchNextStates[count] = self.getPhi(index + 1)
          batchRewards[count]    = self.rewardMemory[index]
          batchActions[count]    = self.actionMemory[index]
          batchNextActions[count]= self.actionMemory[index + 1]
          batchTerminals[count]  = not self.terminalMemory[index + 1]
          batchTasks[count]      = self.taskMemory[index]

          count += 1

        return batchStates, batchActions, batchRewards, batchNextStates, batchNextActions, batchTerminals, batchTasks


    def getLowestSampledTask(self):
        return np.argmin(self.taskSampleCount)

    def __len__(self):
        """ Return the total number of avaible data items. """
        return max(0, min(self.numberOfExperiences - self.phiLength, self.memorySize - self.phiLength))

def main():
    m = AgentMemory((3,2), 4, 10)
    
    for i in xrange(4):
        frame = np.random.randn(3,2)
        m.addFrame(frame)
        m.addExperience(1, 0, False)

    print "StateMemory:"
    print m.stateMemory
    print "\nPhi:\n"
    print m.getPhi()    

    for i in xrange(15):
        frame = np.random.randn(3,2)
        m.addFrame(frame)
        m.addExperience(1, 0, False)


    frame = np.random.randn(3,2)
    m.addFrame(frame)
    m.addExperience(0, 1, True)

    print "\nCurrent Memory Index:" + str(m.currentMemoryIndex) + "\n"

    frame = np.random.randn(3,2)
    m.addFrame(frame)

    print m.stateMemory
    print "\nCurrent Phi crossing end of memory:\n" + str(m.getPhi())

    return m



if __name__ == "__main__":
    main()
