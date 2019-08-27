f_co = open('colors.dat', 'r')
s = f_co.read().split('\n')
f_co.close()


class ColorData:
    def __init__(self,r,g,b, i):
        self.r = r
        self.g = g
        self.b = b
        self.i = i

cor_data = []
cor_name = []
for _ in s:
    __ = _.split(':')
    ___ = __[0].split(',')
    if len(__) == 2 and len(___) == 3:
        cor_data.append(ColorData(float(___[0]), float(___[1]), float(___[2]), int(__[1])))

f_nam = open('names.dat', 'r')
s = f_nam.read().split('\n')
for _ in s:
    cor_name.append(_.split(',')[0])
    print(cor_name[-1])


import numpy as np

test_data_x = []
test_data_y = []
empty_data_y = []

for _ in range(len(cor_name)):
    empty_data_y.append(np.float32(0.0))
    
for _ in cor_data:
    x = []
    x.append(np.float32(_.r))
    x.append(np.float32(_.g))
    x.append(np.float32(_.b))

    y = list(empty_data_y)
    y[_.i] = np.float32(1.0)

    test_data_x.append(x)
    test_data_y.append(y)



#============================================================

import tensorflow as tf


learning_rate = 0.02
num_steps = 1500
display_step = 100

n_hidden_1 = 64
n_hidden_2 = 64
num_input = 3
num_classes = len(cor_name)


X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

batch_x, batch_y = (np.array(test_data_x), np.array(test_data_y))

sess = tf.Session()

sess.run(init)
last_ac = 0
step = 0
while last_ac < 0.9:
    step += 1
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    if step % display_step == 0 or step == 1:
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
        last_ac = acc
        print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Training Accuracy= " + \
            "{:.3f}".format(acc))

print("Optimization Finished!")

print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y}))

while True:
    s = input('r,g,b>>').split(',')
    _ = [float(s[0]), float(s[1]), float(s[2])]
    d = np.argmax(sess.run(logits, feed_dict={X: [_]}))
    print(int(_[0]), int(_[1]) , int(_[2]) ,cor_name[d])
