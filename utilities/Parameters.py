'''
Author Robert Post

Parameters based on code from Nathan Sprague
from: https://github.com/spragunr/deep_q_rl

'''

import argparse
import os
import sys


class Parameters:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000
    ALL_FLAVOR_STRING = ""

    TRANSFER_EXPERIMENT_TYPE = "fullShare"
    DEATH_ENDS_EPISODE = False
    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../ALE/roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    SEED = 33

    IMAGE_WIDTH = 160
    IMAGE_HEIGHT = 210
    REPEAT_ACTION_PROBABILITY = 0.00



    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    CROPPED_WIDTH = 84
    CROPPED_HEIGHT = 84
    LEARNING_RATE = .00025
    DISCOUNT_RATE = .99
    RMS_RHO = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    EPSILON_START = 1.0
    EPSILON_END = .1
    EPSILON_DECAY_STEPS = 1000000
    EVAL_EPSILON = .05
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    MAX_NO_ACTIONS = 30
    BATCH_SIZE = 32
    NETWORK_TYPE = "conv"
    NETWORK_UPDATE_DELAY = 10000
    REPLAY_START_SIZE = 50000
    LOAD_WEIGHTS_FLIPPED = False
    TASK_BATCH_FLAG = 0 # 0 = interleave tasks in 1 batch, 1 = rotate through tasks, 1 task per batch, 2 = randomly select tasks, 1 task per batch
    NUM_HOLDOUT_Q_VALUES = 3200

def processArguments(args, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    """
    defaults = Parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')

    parser.add_argument('--base-rom-path', dest="baseRomPath", default=defaults.BASE_ROM_PATH,
                        help='Where to find the ROMS (default: %(default)s)')

    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="stepsPerEpoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="stepsPerTest",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--cropped-height', dest="croppedHeight",
                        type=int, default=defaults.CROPPED_HEIGHT,
                        help='Desired cropped screen height (default: %(default)s)')
    parser.add_argument('--cropped-width', dest="croppedWidth",
                        type=int, default=defaults.CROPPED_WIDTH,
                        help='Desired cropped screen width (default: %(default)s)')
    parser.add_argument('--merge', dest="mergeFrames", default=False,
                        action="store_true", help='Tell ALE to send the averaged frames')
    parser.add_argument('--display-screen', dest="displayScreen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--frame-skip', dest="frameSkip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process ' + 
                        '(default: %(default)s)')
    parser.add_argument('--update-rule', dest="updateRule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--batch-accumulator', dest="batchAccumulator",
                        type=str, default=defaults.BATCH_ACCUMULATOR,
                        help=('sum|mean (default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learningRate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rmsRho",
                        type=float, default=defaults.RMS_RHO,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rmsEpsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--discount', dest='discountRate', type=float, default=defaults.DISCOUNT_RATE,
                        help='Discount rate')
    parser.add_argument('--epsilon-start', dest="epsilonStart",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-end', dest="epsilonEnd",
                        type=float, default=defaults.EPSILON_END,
                        help='Final epsilon value. (default: %(default)s)')
    parser.add_argument('--eval-epsilon', dest="evalEpsilon",
                        type=float, default=defaults.EVAL_EPSILON,
                        help='Evaluation epsilon value. (default: %(default)s)')
    parser.add_argument('--epsilon-decay-steps', dest="epsilonDecaySteps",
                        type=float, default=defaults.EPSILON_DECAY_STEPS,
                        help=('Number of steps to anneal epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--phi-length', dest="phiLength",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    parser.add_argument('--max-history', dest="replayMemorySize",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batchSize",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--network-type', dest="networkType",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('nips_cuda|nips_dnn|nature_cuda|nature_dnn' +
                              '|linear (default: %(default)s)'))
    parser.add_argument('--network-update-delay', dest="networkUpdateDelay",
                        type=int, default=defaults.NETWORK_UPDATE_DELAY,
                        help=('Interval between target updates. ' +'(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="updateFrequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+ '(default: %(default)s)'))
    parser.add_argument('--replay-start-size', dest="replayStartSize",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' + '(default: %(default)s)'))
    parser.add_argument('--nn-file', dest="nnFile", type=str, default=None,
                        help='Pickle file containing trained net.')
    parser.add_argument('--pause', dest="pause", type=float, default=0,
                        help='Amount of time to pause display while testing.')

    parser.add_argument('--seed', dest="seed", type=int, default=defaults.SEED,
                        help='Seed for ALE.')

    parser.add_argument('--repeat-action-probability', dest="repeatActionProbability", type=float, default=defaults.REPEAT_ACTION_PROBABILITY,
                        help='Repeat Action probability stochasisity in the ALE by randomly using the previous action.')

    parser.add_argument('--loadWeightsFlipped', dest="loadWeightsFlipped", type=int, default=defaults.LOAD_WEIGHTS_FLIPPED,
                        help='Load network conv weights fipped.')

    parser.add_argument('--mode', dest="modeString", type=str, default=defaults.ALL_FLAVOR_STRING,
                        help='String representation of which game modes to use. Either of form x_y, or w,x,y,...,z or x,y;x,z   _ for mode range , for mode separation (single mode or range) and ; to specify different modes for running with multiple games' +
                        '-1 indicates end of mode list')

    parser.add_argument('--difficulty', dest="difficultyString", type=str, default=defaults.ALL_FLAVOR_STRING,
                        help='String representation of which game difficulties to use. Either of form x_y, or w,x,y,...,z or x,y;x,z   _ for diff range , for diff separation (single diff or range) and ; to specify different diffs for running with multiple games'+
                        '-1 indicates end of mode list')

    parser.add_argument('--transferExperimentType', dest="transferExperimentType", type=str, default=defaults.TRANSFER_EXPERIMENT_TYPE, help='String specifying the type of transfer experiment (fullShare|layerShare|representationShare)')
    parser.add_argument('--reduceEpochLengthByNumFlavors', dest="reduceEpochLengthByNumFlavors", default=False, action="store_true", help='Flag to reduce the length of an epoch by the number of flavors')
    parser.add_argument('--evaluationFrequency', dest="evaluationFrequency", type=int, default=1, help=('Evaluation Frequency'))
    parser.add_argument('--useSARSAUpdate', dest="useSARSAUpdate", default=False, action="store_true", help='Flag to set the network target update rule to use a sarsa like update by looking at the next action taken rather than the best action taken for computing q value differences')
    parser.add_argument('--kReturnLength', dest="kReturnLength", type=int, default=1, help='Number of steps to look ahead when computing the return')
    parser.add_argument('--deathEndsEpisode', dest="deathEndsEpisode", default=defaults.DEATH_ENDS_EPISODE, action="store_true", help='Flag to set the loss of life to trigger the end of an episode.')
    parser.add_argument('--maxNoActions', dest="maxNoActions", type=int, default=defaults.MAX_NO_ACTIONS, help='The maximum number of no ops (action 0) that will be executed at the start of an episode')
    parser.add_argument('--taskBatchFlag', dest="taskBatchFlag", type=int, default=defaults.TASK_BATCH_FLAG, help='Flag for transfer module when compiling minibatchs. When 0: tasks are interleaved in a batch. When 1: each batch contains one task and the tasks are rotated through. When 2: each batch contains one tasks and tasks are selected randomly. One task per batch simplifies computation as we dont need to differentiate which sets of parameters to use for convolutions / dot products for each task.')
    parser.add_argument('--numHoldoutQValues', dest="numHoldoutQValues", type=int, default=defaults.NUM_HOLDOUT_Q_VALUES, help='The number of samples used to calculate average Q values over after agent evaluation')

    
    parameters = parser.parse_args(args)
    
    if parameters.networkUpdateDelay > 0:
        parameters.networkUpdateDelay = parameters.networkUpdateDelay // parameters.updateFrequency

    roms = parameters.rom.split(",")
    parameters.roms = []

    for rom in roms:
        if rom == "":
            continue

        newRomPath = os.path.join(parameters.baseRomPath, rom)
        if not parameters.rom.endswith(".bin"):
            newRomPath += ".bin"

        parameters.roms.append(newRomPath)



    return parameters
