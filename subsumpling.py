# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import scipy


parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("resized_dir")
parser.add_argument("output_dir")
args = parser.parse_args()


patch_size = (9, 9) #奇数であること
if patch_size[0] % 2 == 0 or patch_size[1] % 2 == 0:
    print("patch_size should be even")
    exit(1)    

for filename in os.listdir(args.input_dir):
    img = cv2.imread(args.input_dir+"/"+filename)
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    if img.shape == (100, 130):
        first_resized = cv2.resize(img, (20, 26))
        resized = cv2.resize(first_resized, (130, 100))
        cv2.imwrite(args.resized_dir+"/"+filename+".png", resized)

        # 折り返し画像
        large_width = scipy.concatenate((scipy.fliplr(resized), resized, scipy.fliplr(resized)), axis=1)
        large = scipy.concatenate((scipy.flipud(large_width), large_width, scipy.flipud(large_width)), axis=0)

        h, w = resized.shape
        for original_x in xrange(w):
            for original_y in xrange(h):
                x, y = original_x + w, original_y + h
                subimg = large[y-patch_size[1]/2:y+patch_size[1]/2+1, x-patch_size[0]/2:x+patch_size[0]/2+1]
                #cv2.imshow("test", img)
                #cv2.waitKey(-1)
                target_filename="{}_{}_{}.png".format(filename, original_x, original_y)
                cv2.imwrite(args.output_dir+"/"+target_filename, subimg)
                label = float(img[original_y, original_x])/255
                print("{} {}".format(target_filename, label))
