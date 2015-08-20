import sys
from random import randrange
import random
from ale_python_interface import ALEInterface

#
# Given two arrays of difficulty / mode for ale translate between tasks and mode/difficulty combo
# ie a game with 2 difficulties and 3 modes has 6 tasks (0,0), (0, 1), (0, 2), (1,0), (1, 1), (1, 2)
# where a task is a (difficulty, mode) tuple
# 

class TransferTaskModule():
    def __init__(self, taskDifficulties = [0], taskModes = [0]):
        self.taskDifficulties = taskDifficulties
        self.taskModes = taskModes
        self.numTasks = len(self.taskDifficulties) * len(self.taskModes)
        # self.taskIndex = 0

    def getNumTasks(self):
        return self.numTasks


    def getTaskTuple(self, taskIndex): 
        modeIndex = taskIndex % len(self.taskModes)
        diffIndex = taskIndex // len(self.taskDifficulties)

        diff = self.taskDifficulties[diffIndex]
        mod = self.taskModes[modeIndex]

        return (diff,mode)
