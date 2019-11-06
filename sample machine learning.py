f_co = open('colors.dat', 'r')
s = f_co.read().split('\n')
f_co.close()

import pygame
from pygame.locals import *
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)
surf = pygame.display.set_mode((448,400))


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

import tensorflow.compat.v1 as tf

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
while last_ac < 0.97:
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

run = True
r = 255
g = 255
b = 255
_ = [float(r), float(g), float(b)]
d = np.argmax(sess.run(logits, feed_dict={X: [_]}))
text = myfont.render(cor_name[d], False, (255, 255, 255))

while run:
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
            run = False
        elif ev.type == MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if 400 <= x < 416:
                r = int(255 - y / 400 * 255)
            if 416 <= x < 432:
                g = int(255 - y / 400 * 255)
            if 432 <= x < 448:
                b = int(255 - y / 400 * 255)
            _ = [float(r), float(g), float(b)]
            d = np.argmax(sess.run(logits, feed_dict={X: [_]}))
            text = myfont.render(cor_name[d], False, (255, 255, 255))    
            
    surf.fill((0, 0, 0))

    size = text.get_size()
    surf.blit(text,(200 - size[0] // 2,50 - size[1] // 2))
    
    pygame.draw.rect(surf, (r, g, b), (50, 210, 300, 80))

    for _ in range(255):
        pygame.draw.rect(surf, (255 - _, g, b) if r != 255 - _ else (0, 0, 0), (400, 400 / 255 * _, 16, 400 / 255 + 1))
        pygame.draw.rect(surf, (r, 255 - _, b) if g != 255 - _ else (0, 0, 0), (416, 400 / 255 * _, 16, 400 / 255 + 1))
        pygame.draw.rect(surf, (r, g, 255 - _) if b != 255 - _ else (0, 0, 0), (432, 400 / 255 * _, 16, 400 / 255 + 1))
        
    pygame.display.update()
    
    #s = input('r,g,b>>').split(',')
    #_ = [float(s[0]), float(s[1]), float(s[2])]
    #d = sess.run(logits, feed_dict={X: [_]})
    #print(int(_[0]), int(_[1]) , int(_[2]) ,cor_name[d])
