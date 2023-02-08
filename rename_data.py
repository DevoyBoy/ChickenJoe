import os
from os import listdir

folder_dir = "data/data_raw"
counter = 0

for image in os.listdir(folder_dir):
    if image.endswith(".jpg"):
        counter += 1
        # print("old name: " + image)

        # rename image
        if len(image.split('-')) == 2:    # has '-'
            new_name = str(counter) + '-' + image.split('-')[1]
            new_name = new_name.rjust(15, '0')
            x = 1
        else:   
            new_name = str(counter) + image[len(image)-8:]
            new_name = new_name.rjust(14, '0')

        print(new_name)
