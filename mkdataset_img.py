import lmdb
import argparse
import sys
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe
import cv2
import scipy


parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("kvfile")
parser.add_argument("imgdir")
args = parser.parse_args()

db = lmdb.open(path=args.db, map_size=1024*1024*1024)
db.open_db()

with db.begin(write=True) as buf:
    for i, line in enumerate(open(args.kvfile)):
        if i % 10000 == 0: print >> sys.stderr, i
        key = line.split()[0]
        imgfilepath = line.split()[1]
        #img = cv2.imread(args.imgdir+"/"+imgfilepath).transpose([2,0,1])
        img = caffe.io.load_image(args.imgdir+"/"+imgfilepath).astype(float)
        img = scipy.swapaxes(img, 0, 2)
        
        datum = caffe.io.array_to_datum(img)
        buf.put(key, datum.SerializeToString())
