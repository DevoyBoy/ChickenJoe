import os
from os import listdir

folder_dir = "data/new_new_data/"
counter = 9000

for image in os.listdir(folder_dir):
    if image.endswith(".jpg"):
        counter += 1
        # rename image
        if len(image.split('-')) == 2:    # has '-'
            new_name = str(counter) + '-' + image.split('-')[1]
            new_name = new_name.rjust(15, '0')
            x = 1
        else:   
            new_name = str(counter) + image[len(image)-8:]
            new_name = new_name.rjust(14, '0')

        print(image + "\t---> " + new_name)
        # os.rename(folder_dir+image, folder_dir+new_name)
        # TODO: rename image
