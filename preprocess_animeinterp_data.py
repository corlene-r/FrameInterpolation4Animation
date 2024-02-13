from PIL import Image
import os, sys

# I ran this twice; once for test and once for train data

def change_to_png(dir, data_path, filename):
    path = "anim_data/sequences/" + data_path
    i = 0
    for folder in dir:
        if i % 100 == 0: 
            print("\ti: \t", i, "\t out of ", len(dir))
        # To convert the image From JPG to PNG : {Syntax}
        img = Image.open(path + folder + "/frame1.jpg")
        img.save(path + folder + "/im1.png")
        img = Image.open(path + folder + "/frame2.jpg")
        img.save(path + folder + "/im2.png") 
        img = Image.open(path + folder + "/frame3.jpg")
        img.save(path + folder + "/im3.png")
        i = i + 1

    with open('anim_data/' + filename, 'w') as f:
        for folder in dir:
            f.write(data_path + "/" + folder + "\n")

def change_png_dims(dir, path):
    i = 0
    for folder in dir:
        if i % 100 == 0: 
            print("\ti: \t", i, "\t out of ", len(dir))
        # To convert the image From JPG to PNG : {Syntax}
        img = Image.open(path + folder + "/im1.png")
        img = img.resize((448, 256))
        img.save(path + folder + "/im1.png")
        img = Image.open(path + folder + "/im2.png")
        img = img.resize((448, 256))
        img.save(path + folder + "/im2.png")
        img = Image.open(path + folder + "/im3.png")
        img = img.resize((448, 256))
        img.save(path + folder + "/im3.png")
        i = i + 1

path = "anim_data/sequences/test_2k copy/"
data_path = "test_2k copy/"
dir = os.listdir(path)
# print("Change test dims")
# change_png_dims(dir, path)

path = "anim_data/sequences/train_10k/"
data_path = "train_10k/"
dir = os.listdir(path)
# print("Make train pngs")
# change_to_png(dir, data_path, "train list.txt")
print("Change train dim")
change_png_dims(dir, path)

