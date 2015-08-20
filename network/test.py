import theano
import theano.tensor as T
import lasagne
import numpy as np
import TransferLayer

batch_size = 4
numTasks = 4
numUnits = 4
fShape = (2,2)
l_in = lasagne.layers.InputLayer(shape=(batch_size,) + fShape)
transLayer = TransferLayer.TransferLayer(l_in, numTasks, numUnits, True, nonlinearity = None)


transLayerMatricies = transLayer.W.get_value()
transLayerSharedMatrix = transLayer.W_Shared.get_value()

transLayerMatricies[0] = np.zeros((fShape[0] * fShape[1], numUnits))
transLayerMatricies[1] = -1 * np.eye(fShape[0] * fShape[1])
transLayerMatricies[2] = 2 * np.eye(fShape[0] * fShape[1])
transLayerSharedMatrix = 1 * np.eye(fShape[0] * fShape[1])

transLayer.W.set_value(transLayerMatricies)
transLayer.W_Shared.set_value(transLayerSharedMatrix)

x = T.tensor3()
output = lasagne.layers.get_output(transLayer, x)


f = theano.function([x], output)


tasks = [0,1,2,3]
transLayer.setTaskIndices(tasks)


testValues =np.random.randn(batch_size,fShape[0], fShape[1])

print "Task matrix indicies: "
print tasks
print "Task matricies:"
print transLayer.W.get_value()
print "ShareMatrix:"
print transLayer.W_Shared.get_value()
print "Data:"
print testValues
print "Result:"
print f(testValues)


