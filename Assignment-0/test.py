import numpy as np


def softmax(z):
    """ Implement the softmax activation function

    Arguments:
    z - the input of the activation function, shape (BATCH_SIZE, FEATURES) and type `numpy.ndarray`

    Returns:
    a - the output of the activation function, shape (BATCH_SIZE, FEATURES) and type numpy.ndarray
    """

    print("z = " + str(z))
    print("size of z: " + str(z.shape))

    exp_matrix = np.array([np.exp(batch-np.amax(batch)) for batch in z])

    for i, batch in enumerate(exp_matrix):
        for j, feature in enumerate(batch):
            exp_matrix[i, j] = feature/np.sum(batch)


    #exp_matrix = np.exp(z - maximum)
    #sum_e = np.sum(exp_matrix)
    #a = np.array([element / sum_e for element in exp_matrix])
    #print("a =" + str(a))
    a = np.zeros(z.shape)
    print("a :" + str(a))
    return a


test_cases = [
  {
    'input': np.array([[0, 0]]),
    'expected': np.array([[0.5, 0.5]])
  },
  {
    'input': np.array([[1, 1]]),
    'expected': np.array([[0.5, 0.5]])
  },
  {
    'input': np.array([[1.0, 1e-3]]),
    'expected': np.array([[0.7308619, 0.26913807]])
  },
  {
    'input': np.array([[3.0, 1.0, 0.2]]),
    'expected': np.array([[ 0.8360188, 0.11314284, 0.05083836]])
  },
  {
    'input': np.array(
      [[1, 2, 3, 6],
       [2, 4, 5, 6],
       [3, 8, 7, 6]]),
    'expected': np.array(
      [[ 0.006269,  0.01704,  0.04632,  0.93037],
       [ 0.012038,  0.088947,  0.241783,  0.657233],
       [ 0.004462,  0.662272,  0.243636,  0.089629]])
  },
  {
    'input': np.array(
      [[ 0.31323624,  0.7810351 ,  0.26183059,  0.09174578,  0.09806706],
       [ 0.28981829,  0.03154328,  0.99442807,  0.4591928 ,  0.42556593],
       [ 0.06799825,  0.89438807,  0.68276332,  0.89185543,  0.37638809],
       [ 0.49131144,  0.03873597,  0.91306311,  0.2533448 ,  0.24115072],
       [ 0.38297911,  0.23184308,  0.88202174,  0.42546236,  0.78325552]]),
    'expected': np.array(
      [[ 0.19402037,  0.30974891,  0.18429864,  0.15547309,  0.15645899],
       [ 0.16325474,  0.12609515,  0.33027366,  0.19338564,  0.18699081],
       [ 0.11396294,  0.26041151,  0.21074279,  0.25975282,  0.15512995],
       [ 0.21152728,  0.13452883,  0.32250081,  0.16673194,  0.16471114],
       [ 0.16549414,  0.1422804 ,  0.27259261,  0.17267635,  0.2469565 ]])
} ]
for test_case in test_cases:
    ans = softmax(test_case['input'])
    assert np.allclose(ans, test_case['expected'], rtol=1e-07, atol=1e-06), "This test case failed: " + str(test_case)

print('Test passed')