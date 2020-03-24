import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

taraining_inputs = np.array(([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
    ]))

training_outputs = np.array(([
    [0,1,0,1]
    ])).T
np.random.seed(1)

synaptic_weight = 2 * np.random.random((3,1)) - 1

print("Random initialization weights")
print(synaptic_weight)

#back propagation neural network
for i in range(20000):
    firstLayer = taraining_inputs
    outputs = sigmoid( np.dot(firstLayer, synaptic_weight) )

    err = training_outputs - outputs
    adjustments = np.dot( firstLayer.T, err * (outputs * (1 - outputs) ))

    synaptic_weight +=  adjustments

print("Weights after training")
print(synaptic_weight)

print("Result after training")
print(outputs)

