import pylab
import sys
import argparse
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe
import numpy
import random
import os
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("caffemodel")
parser.add_argument("image_name")
parser.add_argument("outimgname")
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
for line in open("train.txt"):
    key, imgpath, r, g, b = line.rstrip().split()
    db[key] = map(lambda x: int(float(x)*256), [b, g, r])


result = numpy.zeros((100, 130, 3))
xys = [(x, y) for x in xrange(130) for y in xrange(100)]

imgs = []
for i,(x,y) in enumerate(xys):
    filepath = "imgs/target/{}_{}_{}.png".format(args.image_name, x, y)
    img = caffe.io.load_image(filepath).astype(float)

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
    # if y==0: print scores * 256
    #print scores.shape
    # if scores[0] != scores[0]: scores[0] = 1 #nan

    r, g, b = map(int, (scores*256).tolist()[0])
    # b, g, r = db["{}_{}_{}.png".format(args.image_name, x, y)]
    result[y, x, 0] = r
    result[y, x, 1] = g
    result[y, x, 2] = b
    #exit(0)

    if i % 100 == 0:
        print i
        #print result
        #cv2.imshow("result", result)
        #cv2.waitKey(-1)
        cv2.imwrite(args.outimgname, result)
