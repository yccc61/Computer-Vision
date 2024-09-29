import numpy as np
import scipy.io
from nn import *
from run_q3 import update_gradient
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']


max_iters = 100
# pick a batch size, learning rate
batch_size = 100
learning_rate =5*1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
#4.3
# initial_W=params["Wlayer1"]
# print(np.shape(initial_W))
# fig = plt.figure(figsize=(10,10))
# grid = ImageGrid(fig, 111,  
#                  nrows_ncols=(8, 8), 
#                  axes_pad=0
#                  )
# for i in range(64):
#     currImg=initial_W[:,i].reshape(32,32)
#     grid[i].imshow(currImg)
# plt.savefig('4.3-1.png')


initialize_weights(hidden_size, 26+10, params, "output")

# with default settings, you should get accuracy > 80%
training_acc=[]
validate_acc=[]
training_cross_entropies=[]

epochs=[i for i in range(max_iters)]
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    
    for xb,yb in batches:
        # forward
        h1=forward(xb,params,name='layer1',activation=sigmoid)
        probs=forward(h1,params,name='output', activation=softmax)
        # loss
        (loss, acc)=compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss+=loss
        total_acc+=acc/len(batches)
        # backward
        delta1 = probs
        y_idx= np.argmax(yb, axis = 1)
        delta1[np.arange(probs.shape[0]),y_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv) 
        # apply gradient
        update_gradient(params,"layer1", learning_rate)
        update_gradient(params,"output", learning_rate)
    
    # run on validation set and report accuracy! should be above 75%
    h1=forward(valid_x, params, name="layer1", activation=sigmoid)
    probs=forward(h1, params, name="output", activation=softmax)
    (loss, valid_acc)=compute_loss_and_acc(valid_y, probs)

    validate_acc.append(valid_acc)
    training_acc.append(total_acc)
    training_cross_entropies.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print('Validation accuracy: ',valid_acc)
    
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



#Q4.1
# plt.plot(epochs, training_acc, label="Training accuracies")
# plt.plot(epochs, validate_acc, label="Validation accuracies")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracies")
# plt.title("Accuracies with 10*best learning rate")
# plt.savefig('4.2-3.png')
# plt.plot(epochs, training_cross_entropies, label="Training entropy")
# plt.xlabel("Epochs")
# plt.ylabel("Entropy")
# plt.title("Entropy with 10*best learning rate")
# plt.savefig('4.2-4.png')
# print("Done")

#Q4.3
# final_W=params["Wlayer1"]
# print(np.shape(final_W))
# fig = plt.figure(figsize=(10,10))
# grid = ImageGrid(fig, 111,  
#                  nrows_ncols=(8, 8), 
#                  axes_pad=0
#                  )
# for i in range(64):
#     currImg=final_W[:,i].reshape(32,32)
#     grid[i].imshow(currImg)
# print("Done")
# plt.savefig('4.3-2.png')

# Q4.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
h1=forward(valid_x, params, name="layer1", activation=sigmoid)
probs=forward(h1, params, name="output", activation=softmax)
prediction=np.argmax(probs, axis=1)
y_label=np.argmax(valid_y, axis=1)
for i in range(len(prediction)):
    confusion_matrix[prediction[i], y_label[i]]+=1
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.savefig("Q4.4.png")
