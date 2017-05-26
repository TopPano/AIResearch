import numpy as np
from caffe2.python import cnn, workspace
from caffe2.python import net_drawer

def gen_graph(net, output):
    graph = net_drawer.GetPydotGraph(net, rankdir = 'LR')
    graph.write_png(output)

# Create input data
data = np.random.rand(16, 100).astype(np.float32)
# Create Labels for the data as integers [0, 9]
label = (np.random.rand(16) * 10).astype(np.int32)

# Feed data and label
workspace.FeedBlob('data', data)
workspace.FeedBlob('label', label)

# Create a model using model helper
m = cnn.CNNModelHelper(name = 'hello caffe2')
fc_1 = m.FC('data', 'fc1', dim_in = 100, dim_out = 10)
pred = m.Sigmoid(fc_1, 'pred') # or: m.Sigmoid('fc_1', 'pred')
[softmax, loss] = m.SoftmaxWithLoss([pred, 'label'], ['softmax', 'loss'])
m.AddGradientOperators([loss])

# Debug nets via printing
print(str(m.net.Proto()))
print(str(m.param_init_net.Proto()))

# Debug nets via visulizing
gen_graph(m.net, 'net.png')
gen_graph(m.param_init_net, 'param_init_net.png')

# Parameters initialization
workspace.RunNetOnce(m.param_init_net)

# You can print out if you are interested
# print(workspace.FetchBlob('fc1_b'))
# print(workspace.FetchBlob('fc1_w'))

# Create the actual training net
workspace.CreateNet(m.net)

# Run 100 iterations
for j in range(100):
    workspace.RunNet(m.name, 10) # run for 10 times

    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob('data', data)
    workspace.FeedBlob('label', label)

    # You can print them out if you are interested in
    # print(workspace.FetchBlob('softmax'))
    # print(workspace.FetchBlob('loss'))
