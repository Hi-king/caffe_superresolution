import lmdb
import argparse
import sys
sys.path.insert(0, "/home/ogaki/Workspace/caffe/python")
import caffe


parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("kvfile")
args = parser.parse_args()

db = lmdb.open(args.db)
db.open_db()

with db.begin(write=True) as buf:
    for line in open(args.kvfile):
        key, value = line.split()
        value = float(value)
        data = caffe.io.caffe_pb2.Datum()
        data.float_data.extend([value])
        data.channels, data.height, data.width = 1, 1, 1
        print data
        buf.put(key, data.SerializeToString())
