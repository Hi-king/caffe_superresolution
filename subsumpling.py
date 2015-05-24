# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import scipy
import random

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("resized_dir")
parser.add_argument("output_dir")
args = parser.parse_args()


patch_size = (9, 9) #奇数であること
if patch_size[0] % 2 == 0 or patch_size[1] % 2 == 0:
    print("patch_size should be even")
    exit(1)    


key = 0
for filename in os.listdir(args.input_dir):
    img = cv2.imread(args.input_dir+"/"+filename)
    # img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    if img.shape == (100, 130, 3):
        first_resized = cv2.resize(img, (65, 50))
        resized = cv2.resize(first_resized, (130, 100))
        cv2.imwrite(args.resized_dir+"/"+filename+".png", resized)

        # 折り返し画像
        large_width = scipy.concatenate((scipy.fliplr(resized), resized, scipy.fliplr(resized)), axis=1)
        large = scipy.concatenate((scipy.flipud(large_width), large_width, scipy.flipud(large_width)), axis=0)

        h, w, d = resized.shape
        xys = [(x, y) for x in xrange(w) for y in xrange(h)]
        random.shuffle(xys)
        for original_x, original_y in xys:
            key += 1
            x, y = original_x + w, original_y + h
            subimg = large[y-patch_size[1]/2:y+patch_size[1]/2+1, x-patch_size[0]/2:x+patch_size[0]/2+1]
            #cv2.imshow("test", img)
            #cv2.waitKey(-1)
            target_filename="{}_{}_{}.png".format(filename, original_x, original_y)
            cv2.imwrite(args.output_dir+"/"+target_filename, subimg)
            label = map(lambda item: float(item)/256, img[original_y, original_x].tolist())
            print(" ".join(map(str, [key, target_filename]+label)))
