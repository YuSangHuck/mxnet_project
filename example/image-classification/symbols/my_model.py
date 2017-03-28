# it doesn't work
import mxnet as mx

def get_symbol(num_classes=10, **kwargs):
	input_data = mx.symbol.Variable(name='data')
	conv1 = mx.symbol.Convolution(name='conv1',data=input_data,kernel=(3,3),stride=(1,1),num_filter=64)
	relu1 = mx.symbol.Activation(data=conv1,act_type='relu')
	pool1 = mx.symbol.Pooling(data=relu1,pool_type='max',kernel=(3,3),stride=(2,2))
	conv2 = mx.symbol.Convolution(name='conv1',data=input_data,kernel=(3,3),stride=(1,1),num_filter=64)
	relu2 = mx.symbol.Activation(data=conv1,act_type='relu')
	pool2 = mx.symbol.Pooling(data=relu1,pool_type='max',kernel=(3,3),stride=(2,2))
	
	flatten = mx.symbol.Flatten(data=pool2)
	
	fc1 = mx.symbol.FullyConnected(name='fc1',data=flatten,num_hidden=256)
	relu3 = mx.symbol.Activation(data=fc1,act_type='relu')
	dropout1 = mx.symbol.Dropout(data=relu3,p=0.5)
	fc2 = mx.symbol.FullyConnected(name='fc2',data=dropout1,num_hidden=128)
	relu4 = mx.symbol.Activation(data=fc2,act_type='relu')
	dropout2 = mx.symbol.Dropout(data=relu4,p=0.5)
	fc3 = mx.symbol.FullyConnected(name='fc3',data=dropout2,num_hidden=num_classes)
	softmax = mx.symbol.SoftmaxOutput(data=fc3,name='softmax')

	return softmax	
