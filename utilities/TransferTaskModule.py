import sys
from random import randrange
import random
from ale_python_interface import ALEInterface
import Environment

#
# Given a list of games and two arrays of difficulty / mode for ale translate between tasks and game/mode/difficulty combo
# ie a game with 2 difficulties and 3 modes has 6 tasks (0,0), (0, 1), (0, 2), (1,0), (1, 1), (1, 2)
# where a task is a (game, difficulty, mode) tuple
# 

def createFlavorList(flavorString, availableFlavors):
    #Creates a list of number out of string for selecting modes/difficulties from a string parameter
    flavorList = []
    for s in flavorString.split(','):
        numRange = s.split('_')
        if len(numRange) == 1:
            flavorList.append(availableFlavors[int(s)])
        else:
            start = int(numRange[0])
            end = int(numRange[-1])
            if end == -1:
                end = len(availableFlavors) - 1
            for num in range(start, end + 1):
                flavorList.append(availableFlavors[num])
    return list(set(flavorList))

# def createFlavorList(flavorString, maxNumFlavors):
#     #Creates a list of number out of string for selecting modes/difficulties from a string parameter
#     flavorList = []
#     for s in flavorString.split(','):
#         numRange = s.split('_')
#         if len(numRange) == 1:
#             flavorList.append(int(s))
#         else:
#             start = int(numRange[0])
#             end = int(numRange[-1])
#             if end == -1:
#                 end = maxNumFlavors - 1
#             for num in range(start, end + 1):
#                 flavorList.append(num)
#     return list(set(flavorList))

class TransferTaskModule():
    def __init__(self, ale, gameList, difficultyString= "", modeString= "", taskBatchFlag = 0):
        self.ale = ale
        self.gameList = gameList
        self.gameInfo = []
        self.taskBatchFlag = taskBatchFlag
        
        #Default to 0th difficulty and mode for all games when not specified
        if difficultyString == "":
            numGames = len(gameList)
            for x in xrange(numGames):
                difficultyString += "0;"
            difficultyString = difficultyString[0:-1]

        if modeString == "":
            numGames = len(gameList)
            for x in xrange(numGames):
                modeString += "0;"
            modeString = modeString[0:-1]


        difficultyStringList = difficultyString.split(";")
        modeStringList = modeString.split(";")

        gameCounter = 0 
        self.numTasks = 0
        self.allActionsList = set()

        for game in self.gameList:
            self.gameInfo.append({"gamePath":game})
            ale.loadROM(game)
            self.gameInfo[gameCounter]["minActions"] = ale.getMinimalActionSet()
            self.allActionsList = self.allActionsList.union(set(self.gameInfo[gameCounter]["minActions"]))

            availableDiffs = ale.getAvailableDifficulties()
            availableModes = ale.getAvailableModes()

            self.gameInfo[gameCounter]["difficulties"] = createFlavorList(difficultyStringList[gameCounter], availableDiffs)
            self.gameInfo[gameCounter]["modes"] = createFlavorList(modeStringList[gameCounter], availableModes)
            self.gameInfo[gameCounter]["numFlavors"] = len(self.gameInfo[gameCounter]["difficulties"]) * len(self.gameInfo[gameCounter]["modes"])
            self.numTasks += self.gameInfo[gameCounter]["numFlavors"]
            gameCounter += 1

        self.allActionsList = sorted(list(self.allActionsList))
        

        for game in self.gameInfo:
            game["actionIndices"] = []
            for action in game["minActions"]:
                if action in self.allActionsList:
                    game["actionIndices"].append(self.allActionsList.index(action))

        self.currentGameIndex = 0
        self.currentTaskIndex = 0 

    def getNumTasks(self):
        return self.numTasks

    def getNumGames(self):
        return len(self.gameInfo)

    def getNumTotalActions(self):
        return len(self.allActionsList)

    def getTotalActionsList(self):
        return self.allActionsList

    def getActionsForCurrentTask(self):
        return self.gameInfo[self.currentGameIndex]["actionIndices"]

    def getTaskTuple(self, taskIndex = None):
        if taskIndex == None:
            taskIndex = self.currentTaskIndex
        
        assert taskIndex >= 0

        taskCounter = taskIndex
        currentGame = None
        
        for i in xrange(0, len(self.gameInfo)):
            game = self.gameInfo[i]
            if taskCounter < game["numFlavors"]:
                currentGame = game
                currentGameIndex = i
                break
            
            taskCounter -= game["numFlavors"]

        assert(currentGame != None)


        modeIndex = taskCounter % len(currentGame["modes"])
        diffIndex = taskCounter // len(currentGame["modes"])

        newGamePath = currentGame["gamePath"]
        diff = currentGame["difficulties"][diffIndex]
        mode = currentGame["modes"][modeIndex]

        return (newGamePath, diff,mode, currentGameIndex)

    def changeToTask(self, newTaskNumber):
        assert newTaskNumber >= 0 and newTaskNumber < self.getNumTasks()

        rom,diff,mode, currentGameIndex = self.getTaskTuple(newTaskNumber)
        self.currentTaskIndex = newTaskNumber
        self.currentGameIndex = currentGameIndex
        self.ale.loadROM(rom)
        self.ale.setMode(mode)
        self.ale.setDifficulty(diff)



def test():
    ale = ALEInterface()
    Environment.initializeALEParameters(ale, 1, 4, 0.00, False)
    # t = TransferTaskModule(ale, ["../../ALE/roms/boxing.bin", "../../ALE/roms/hero.bin", "../../ALE/roms/space_invaders.bin"], "0;0_-1;0_-1", "0;0_-1;0_-1")
    t = TransferTaskModule(ale, ["../../ALE/roms/pong.bin"], "0_-1", "0_-1")
    for x in range(t.getNumTasks()):
        print t.getTaskTuple(x)

    return t

