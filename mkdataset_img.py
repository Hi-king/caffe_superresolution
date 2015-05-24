import lmdb
import argparse
import sys
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("kvfile")
parser.add_argument("imgdir")
args = parser.parse_args()

db = lmdb.open(args.db)
db.open_db()

with db.begin(write=True) as buf:
    for i, line in enumerate(open(args.kvfile)):
        if i % 10000 == 0: print >> sys.stderr, i
        key, value = line.split()
        img = cv2.imread(args.imgdir+"/"+key).transpose([2,0,1])

        datum = caffe.io.array_to_datum(img)
        buf.put(key, datum.SerializeToString())
