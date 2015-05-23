import pylab
import sys
import argparse
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe
import numpy
import os
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("caffemodel")
parser.add_argument("image_name")
parser.add_argument("outdir")
args = parser.parse_args()

caffe.set_phase_test()
caffe.set_mode_cpu()
network = caffe.Classifier(
        "./deploy.prototxt",
        args.caffemodel
    )
#network.set_mean('data', numpy.load("/home/ogaki/Workspace/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"))
#network.set_raw_scale('data', 255)
#network.set_channel_swap('data', (2,1,0))


result = numpy.zeros((100, 130))
for x in xrange(10):
    print >> sys.stderr, x
    for y in xrange(100):
        filepath = "imgs/target/{}_{}_{}.png".format(args.image_name, x, y)
        scores = network.predict([caffe.io.load_image(filepath)])
        #print scores
        #print scores.shape
        #print scores.argmax()
        result[y, x] = scores.argmax()
print result
cv2.imshow("result", result)
cv2.waitKey(-1)
cv2.imwrite("result.png", result)
    
