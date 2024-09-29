import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 3.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    
    b=[0 for i in range (out_size)]
    b=np.array(b)

    #Uniform distribution variance=1/b-a
    #variance=2/(in_size+out_size)
    #let 1/12*(b-a)^2
    # W=np.random.uniform(low=0, high=np.sqrt(24/(in_size+out_size)), size=(in_size, out_size))
    W=np.random.uniform(low=-np.sqrt(6/(in_size+out_size)), high=np.sqrt(6/(in_size+out_size)),size=(in_size, out_size))
    
    params['W' + name] = W
    params['b' + name] = b

# Q 3.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 3.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act=X@W+b
    post_act=activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 3.2.2
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    #normalize x since softmax is invariant to translation
    c=np.reshape(np.max(x, axis=1),(-1,1))
    x=x-c
    exp_x=np.exp(x)
    sum_x=np.reshape(np.sum(exp_x, axis=1),(-1,1))
    res = exp_x/sum_x
    return res

# Q 3.2.3
# compute average loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    
    (D, classes)=np.shape(y)
    # Loss= -1/D* np.sum(np.log(probs[y==1]))
    loss, acc = None, None
    loss= -1/D * np.sum(y*np.log(probs))
    label=np.argmax(y, axis=1)
    prediction=np.argmax(probs, axis=1)
    acc=np.sum(label==prediction)/D
    
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 3.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    
    # do the derivative through activation first
    #compute partial J/partial z by multiplying partial J/partial y * partial y/partial z
    delta*=activation_deriv(post_act)
    # then compute the derivative W,b, and X
    grad_X=delta@np.transpose(W)
    grad_W=np.transpose(X)@delta
    grad_b=np.ones(len(delta))@delta

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 3.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches=[]
    num_examples=np.shape(x)[0]
    shuffled_index=np.random.permutation(num_examples)
    num_batches=num_examples//batch_size
    for i in range(num_batches):
        start=i*batch_size
        end=(i+1)*batch_size
        index=shuffled_index[start:end]
        curr_batch_x=x[index, :]
        curr_batch_y=y[index, :]
        batches+=[(curr_batch_x, curr_batch_y)]
        

    return batches
