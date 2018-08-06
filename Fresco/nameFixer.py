import glob
import os

pic_dir = "/home/draco/jthornton/FrescoPictures/"
clustername = "DeltaF2/"

search = glob.glob(pic_dir+clustername+"*.png")
for name in search:
    print name
    print name[:-6]+".png"
    os.system("mv "+name+" "+name[:-6]+".png")
