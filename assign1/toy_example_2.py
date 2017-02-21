import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def non_linearity(X, type='None'):

      if type == 'None':
          return X

      dim = X.size
      response = np.zeros_like(X)

      if type =='Sigmoid':
            response = 1/(1+np.exp(-X))

      if type == 'Sign':
        for i in xrange(dim):
            if X[i]>0:
                response[i] = 1
            else:
                response[i] = -1

      return response


def sigmoid_gradient(sigmX):
    return sigmX*(1-sigmX)


class Layer(object):

  def __init__(self):
      self.num_units = None
      self.W = None
      self.node_number = None
      self.non_linearity = None
      self.linear_input = None
      self.linear_response = None
      self.nonlinear_input = None
      self.nonlinear_response = None
      self.linear_gradient = None
      self.non_linear_gradient = None
      self.sensitivity = None

  def set_sensitivity(self, x):
      self.sensitivity = x
      return

  def get_sensitivity(self):
      return self.sensitivity

  def layer_details(self):
      print 'Layer number:', self.layer_number
      print 'Number of units:', self.num_units
      print 'Non-linearity:', self.non_linearity
      print 'W:', self.W

  def apply_linear_opeartion(self, X):
      response = self.W.dot(X)
      self.linear_input = X
      self.linear_response = response
      return response

  def apply_nonlinear_operation(self, X):
      self.nonlinear_input = X
      response = non_linearity(X, self.non_linearity)
      self.nonlinear_response = response
      return response

  def compute_non_linear_gradient(self):
      if self.non_linearity =='None':
        self.non_linear_gradient = self.nonlinear_input
      else:
        self.non_linear_gradient = sigmoid_gradient(self.nonlinear_response)
      return self.non_linear_gradient

  def compute_linear_gradient(self):
      self.linear_gradient = self.linear_input
      return self.linear_gradient

# 3-4-1-1
X = np.array([
                [0,0],
                [0,1],
                [1,0],
                [1,1]
])
Y = np.array([[0,1,1,0]]).T

'''
X = np.array([
                [0,1,1],
                [1,0,0],
                [0,1,0],
                [0,0,1],
                [1,1,0],
                [1,0,1]
            ])
Y = np.array([[1,0,0,0,1,1]]).T
4-10-4-1
'''

X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

num_samples = X[:,0].size

# network architecture
num_layers = 3
layer_objects = []

for i in xrange(num_layers):
    layer_objects.append(Layer())

#layer1 = input layer
layer_objects[0].layer_number = 1
layer_objects[0].num_units = 3
layer_objects[0].W = np.eye(3, 3)
layer_objects[0].dW = np.zeros_like(layer_objects[0].W)
layer_objects[0].non_linearity = 'None'

#layer 2
layer_objects[1].layer_number = 2
layer_objects[1].num_units = 4
layer_objects[1].W = ((2/np.sqrt(4))*np.random.random((4, 3)))-1/np.sqrt(4)
layer_objects[1].dW = np.zeros_like(layer_objects[1].W)
layer_objects[1].non_linearity = 'Sigmoid'

#layer 3
layer_objects[2].layer_number = 3
layer_objects[2].num_units = 1
layer_objects[2].W = W = ((2/np.sqrt(2))*np.random.random((1, 4)))-1/np.sqrt(2)
layer_objects[2].dW = np.zeros_like(layer_objects[2].W)
layer_objects[2].non_linearity = 'Sigmoid'

#layer 4
#layer_objects[3].layer_number = 4
#layer_objects[3].num_units = 1
#layer_objects[3].W = W = ((2/np.sqrt(1))*np.random.random((1, 1)))-1/np.sqrt(1)
#layer_objects[3].dW = np.zeros_like(layer_objects[3].W)
#layer_objects[3].non_linearity = 'Sigmoid'

num_iters = 10000
loss_batch = np.zeros((num_iters,1))

for i in xrange(num_iters):

    loss = 0

    for layer_id in xrange(num_layers):
        layer_objects[layer_id].dW = np.zeros_like(layer_objects[layer_id].W)

    for j in xrange(num_samples):

        input = X[j]
        # forward pass
        for layer_id in xrange(num_layers):
            node = layer_objects[layer_id]
            activations = node.apply_linear_opeartion(input)
            output = node.apply_nonlinear_operation(activations)
            input = output
        loss += 0.5*np.square(Y[j]-output)

        #backward pass
        for layer_id in xrange(num_layers-1, -1, -1):

             node = layer_objects[layer_id]
             grad_nl = node.compute_non_linear_gradient()
             grad_l = node.compute_linear_gradient()

             if layer_id == num_layers-1:
                sensitivity = -1*(Y[j]-output)*np.array([grad_nl])
             else:
                previous_node = layer_objects[layer_id+1]
                A = previous_node.get_sensitivity()
                B = np.dot(previous_node.W.T, A.T)
                sensitivity = B.T*grad_nl

             node.set_sensitivity(sensitivity)
             node.dW += sensitivity.T*grad_l

    for layer_id in xrange(num_layers):
        layer_objects[layer_id].W = layer_objects[layer_id].W  - 0.1*layer_objects[layer_id].dW

    loss/=num_samples
    loss_batch[i] = loss
    if i%100==0:
        print "Loss after iteration", i, ':', loss

# Testing
for j in xrange(num_samples):
      input = X[j]
      for layer_id in xrange(num_layers):
            node = layer_objects[layer_id]
            activations = node.apply_linear_opeartion(input)
            output = node.apply_nonlinear_operation(activations)
            input = output
      print Y[j], output

plt.interactive(False)
plt.plot(loss_batch)
plt.show()