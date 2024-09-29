import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *


# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])

# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}

# Q 3.1.2
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))

#expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(),params['Woutput'].std()**2))

# Q 3.2.1
# implement sigmoid
test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())

# implement forward
h1 = forward(x,params,'layer1')
print(h1.shape)
# Q 3.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# positive, ~1, ~1, (40,4)
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 3.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around 1.65 and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1

# Q 3.3.1
# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 3.5.1 should be implemented in this file
# you can do this before or after training the network. 
# CODE TO WRITE: 
#       You should do a forward & backward pass of the dataset here to get
#       params populated with a gradient you expect (but do not apply it)


# save the old params
# now we want to save the gradients that you just computed
import copy
params_orig = copy.deepcopy(params)

# now we'll try to get the same result with numerical gradients
eps = 1e-6
for k,v in params.items():
    if '_' in k: 
        continue
    # we have a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   compute derivative with central diffs
    #   store that inside params

    # you can just fill this out
    grad_v = np.zeros_like(v)
    if len(np.shape(grad_v))==1:
        for i in range(np.shape(grad_v)[0]):
            params[k][i]+=eps
            h1=forward(x,params,name='layer1',activation=sigmoid)
            probs=forward(h1,params,name='output', activation=softmax)
            (loss1, acc)=compute_loss_and_acc(y, probs)
            params[k][i]-=eps*2
            h1=forward(x,params,name='layer1',activation=sigmoid)
            probs=forward(h1,params,name='output', activation=softmax)
            (loss2, acc)=compute_loss_and_acc(y, probs)
            params[k][i]+=eps
            grad_v[i]=(loss1-loss2)/(2*eps)
    else: 
        for i in range(np.shape(grad_v)[0]):
            for j in range(np.shape(grad_v)[1]):
                params[k][i,j]+=eps
                h1=forward(x,params,name='layer1',activation=sigmoid)
                probs=forward(h1,params,name='output', activation=softmax)
                (loss1, acc)=compute_loss_and_acc(y, probs)
                params[k][i,j]-=eps*2
                h1=forward(x,params,name='layer1',activation=sigmoid)
                probs=forward(h1,params,name='output', activation=softmax)
                (loss2, acc)=compute_loss_and_acc(y, probs)
                params[k][i,j]+=eps
                grad_v[i,j]=(loss1-loss2)/(2*eps)
    params['grad_' + k] = grad_v

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print(err)
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
params=params_orig


# Q 3.4.1

def update_gradient(params,name, learning_rate):
    W = params['W' + name]
    b = params['b' + name]

    grad_W=params['grad_W' + name]
    grad_b=params['grad_b' + name]

    params['W'+name]=W-learning_rate*grad_W
    params['b'+name]=b-learning_rate*grad_b

batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# WRITE A TRAINING LOOP HERE
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get accuracy > 75%

for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        # forward
        h1=forward(xb,params,name='layer1',activation=sigmoid)
        probs=forward(h1,params,name='output', activation=softmax)
        # loss
        (loss, acc)=compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss+=loss
        avg_acc+=acc/len(batches)
        # backward
        delta1 = probs
        y_idx= np.argmax(yb, axis = 1)
        delta1[np.arange(probs.shape[0]),y_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv) 
        # apply gradient
        update_gradient(params,"layer1", learning_rate)
        update_gradient(params,"output", learning_rate)
        
        
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


