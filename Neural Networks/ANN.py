"""
@author: diti
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #importing our mnist data

#dataset of images of handwritten digits in black and white from 0-9
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True) #the data path is the first parameter, 
#second parameter is set to true for activating an output node whenever we get 
#the corresponding output for example for 0=[1 0 0 0 0 0 0 0 0 0],1=[0 1 0 0 0 0 0 0 0 0] so on

#define the total number of nodes in each layer
h1_nodes=500#note that the total nodes and hidden layers can vary as you want
h2_nodes=600#more the number of hidden layers, deeper the neural network
h3_nodes=300

#define total number of output classes
n_classes=10 #here 10 as there are 10 digits as output classes(0-9)

#define batch_size
batch_size=100 #a batch of 100 would be processed together to the net

#define the input x and output y
x=tf.placeholder('float64',[None,784])
 #x is matrix here of pixels of images with dimensions 28*28 pixels 
 #placeholder() is function with parameters dtype,shape and name
 #Here dtype=float, shape(optional)=matrix of 28*28 image and name of process is optional
y=tf.placeholder('float64')

#define the model
def ann_model(data):
    data=tf.cast(data,tf.float32)
    #define random weights and biases for each hidden layer:
    hidden1={'weights':tf.Variable(tf.random_normal([784, h1_nodes])),
             'biases':tf.Variable(tf.random_normal([h1_nodes]))}
    
    hidden2={'weights':tf.Variable(tf.random_normal([h1_nodes, h2_nodes])),
             'biases':tf.Variable(tf.random_normal([h2_nodes]))}
    
    hidden3={'weights':tf.Variable(tf.random_normal([h2_nodes, h3_nodes])),
             'biases':tf.Variable(tf.random_normal([h3_nodes]))}
    
    output={'weights':tf.Variable(tf.random_normal([h3_nodes, n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    #matrix multiplication to feed to activation function
    
    l1=tf.add(tf.matmul(data,hidden1['weights']), hidden1['biases'])#sum of (input*weight+bias)
    l1=tf.nn.relu(l1)#activation function
    
    l2=tf.add(tf.matmul(l1,hidden2['weights']), hidden2['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden3['weights']), hidden3['biases'])
    l3=tf.nn.relu(l3)
    
    output_layer=tf.add(tf.matmul(l3,output['weights']), output['biases'])#output layer has matrix multipication only
    
    return output_layer

#define the training function of model
def train_model(x,y):
    prediction=ann_model(x)#predicting using model
    #reducing cost by using cross entropy between predicted value and expected value y
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    
    optimizer=tf.train.AdamOptimizer().minimize(cost)#optimize the output by minimizing the cost
    
    epochs=10 #total epochs
    #start the session/looping/working
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#initializing all variables
        for epoch in range(epochs):
            e_loss=0#epoch loss initialize
            #dividing our total training examples into batches
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)#training
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})#running optimizer and cost calculation
                #feed_dict is used to feed x,y values to the respective placeholders x,y
                e_loss=+c #epoch loss 
            print('Epoch ',epoch,' completed out of ',epochs,' with loss:',e_loss)
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#to check if one_hot of prediction is identical to that of expected output
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))#evaluate accuracy

#running the model
train_model(x,y)