import lmdb
import argparse
import sys
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe


parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("kvfile")
args = parser.parse_args()

db = lmdb.open(path=args.db, map_size=1024*1024*1024)
db.open_db()

with db.begin(write=True) as buf:
    for line in open(args.kvfile):
        key, imgpath, r, g, b = line.split()
        r, g, b = float(r), float(g), float(b)
        data = caffe.io.caffe_pb2.Datum()
        data.float_data.extend([r, g, b])
        data.channels, data.height, data.width = 3, 1, 1
        buf.put(key, data.SerializeToString())
