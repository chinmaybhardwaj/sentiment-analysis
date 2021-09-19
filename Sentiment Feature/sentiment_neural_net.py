import tensorflow as tf
from create_sentiment_features import create_feature_sets_and_labels
import numpy as np

train_X, train_y, test_X, test_y = create_feature_sets_and_labels('./dataset/pos.txt', './dataset/pos.txt')

# Hidden layers and nodes
n_nodes_hidden_l1 = 500
n_nodes_hidden_l2 = 500
n_nodes_hidden_l3 = 500


n_classes = 2
batch_size = 100

X = tf.placeholder('float',[None, len(train_X[0])])
y = tf.placeholder('float')


def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_X[0]), n_nodes_hidden_l1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hidden_l1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_l1, n_nodes_hidden_l2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hidden_l2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_l2, n_nodes_hidden_l3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hidden_l3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_l3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    
    # (input_data * weights) + biases
    l1 = tf.add( tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add( tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add( tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output



def train_neural_network(X):
    # Get output from neural network
    prediction = neural_network_model(X)
    # Get cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Cycles of feed forward + backpropagation
    n_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training data
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_X):
                start = i
                end = i + batch_size
                
                batch_X = np.array(train_X[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer,cost], feed_dict={X: batch_X, y: batch_y}) # c = cost
                epoch_loss += c
                i += batch_size
            print('Epoch:', epoch + 1, 'completed out of', n_epochs, 'loss:', epoch_loss)
             
        
        
        # 
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X:test_X, y: test_y}))



train_neural_network(X)

