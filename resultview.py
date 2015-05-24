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



db = {}
for line in open("train_float.txt"):
    key, value = line.rstrip().split()
    db[key] = int(float(value)*256)

result = numpy.zeros((100, 130, 3))
for x in xrange(130):
    print >> sys.stderr, x
    for y in xrange(100):
        filepath = "imgs/target/{}_{}_{}.png".format(args.image_name, x, y)
        img = caffe.io.load_image(filepath)
        #import cv2
        #cv2.imshow("img", img)
        #cv2.waitKey(-1)
        #print img.shape
        #print img.max()
        #print img
        scores = network.predict([img])
        #print network.blobs
        #print network.blobs['ip1'].data
        #print network.blobs['ip1'].data.shape
        if y==0: print scores * 256
        #print scores.shape
        if scores[0] != scores[0]: scores[0] = 1 #nan
        power = int(scores[0]*256)
        result[y, x, 0] = power
        result[y, x, 1] = power
        result[y, x, 2] = power
        #exit(0)

    if x % 10 == 0:
        #print result
        #cv2.imshow("result", result)
        #cv2.waitKey(-1)
        cv2.imwrite("result/result.png", result)
