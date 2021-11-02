##0
# Let's use what we already know about deployment to run an image through this model. Start by initializing the model:
import caffe
import numpy as np
#caffe.set_mode_gpu()
import matplotlib.pyplot as plt #matplotlib.pyplot allows us to visualize results

ARCHITECTURE = 'deploy.prototxt' #a locally stored version
WEIGHTS = 'bvlc_alexnet.caffemodel'#a locally stored version
MEAN_IMAGE = 'ilsvrc_2012_mean.npy'
#TEST_IMAGE = '/dli/data/BeagleImages/louietest2.JPG'
TEST_IMAGE = 'starsky.jpg'

# Initialize the Caffe model using the model trained in DIGITS
net = caffe.Classifier(ARCHITECTURE, WEIGHTS) #Each "channel" of our images are 256 x 256

##1
# Then create an input the network expects. Note that this is different than the preprocessing used in the last model.
# To learn how imagenet was preprocessed, the documentation was clearly presented on http://caffe.berkeleyvision.org/gathered/examples/imagenet.html
#Load the image
image= caffe.io.load_image(TEST_IMAGE)
plt.imshow(image)
plt.show()

#Load the mean image
mean_image = np.load(MEAN_IMAGE)
mu = mean_image.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# set the size of the input (we can skip this if we're happy with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

transformed_image = transformer.preprocess('data', image)

##2
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output

##3
#Work to make the output useful to a user.
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print('predicted class is:', output_prob.argmax())

labels_file = 'synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print('output label:', labels[output_prob.argmax()])

print("Input image:")
plt.imshow(image)
plt.show()

print("Output label:" + labels[output_prob.argmax()])