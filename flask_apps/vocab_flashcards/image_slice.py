import os
import sys
import image_slicer
from PIL import Image
from shutil import copyfile, rmtree


def slicer():
    directory_in_str = '/Users/travis.howe/Desktop/rc/vocab_images'

    for file in os.listdir(directory_in_str):
        tiles = image_slicer.slice('{0}/{1}'.format(directory_in_str, file), 8, save=False)
        image_slicer.save_tiles(tiles, directory='{0}_sliced'.format(directory_in_str), prefix=file)


def renamer():
    directory_num = list(range(4, 13)) + list(range(14, 76))
    for num in directory_num:
        directory = '/Users/travis.howe/Desktop/rc/vocab_images_chapters/{0}'.format(num)

        destination_directory = '/Users/travis.howe/Projects/github/data_science/flask_apps/vocab_flashcards/static/{}'.format(num)
        if not os.path.exists(destination_directory):
            print('Directory does not exist...creating it.')
            os.makedirs(destination_directory)
        else:
            print('Directory already exists...removing and then creating it anew.')
            rmtree(destination_directory)
            os.makedirs(destination_directory)

        for ind, file in enumerate(os.listdir(directory)):
            copyfile('{0}/{1}'.format(directory, file), '{0}/rc_vocab_{1}_{2}.png'.format(destination_directory, num, ind))


def image_trim():
    directory_num = list(range(4, 13)) + list(range(14, 150))
    for num in directory_num:
        print("Folder: ", num)
        directory = '/Users/travis.howe/Desktop/rc/vocab_images_chapters/{0}'.format(num)

        destination_directory = '/Users/travis.howe/Projects/github/data_science/flask_apps/vocab_flashcards/static/{}'.format(num)
        if not os.path.exists(destination_directory):
            print('Directory does not exist...creating it.')
            os.makedirs(destination_directory)
        else:
            print('Directory already exists...removing and then creating it anew.')
            rmtree(destination_directory)
            os.makedirs(destination_directory)

        for ind, file in enumerate(os.listdir(directory)):
            # copyfile('{0}/{1}'.format(directory, file), '{0}/rc_vocab_{1}_{2}.png'.format(destination_directory, num, ind))

            # filename = '/Users/travis.howe/Desktop/rc/vocab_images_chapters/4/VocFC_502_-_Vocabulary_Flash_Cards_1-80-01.png_01_01.png'
            im = Image.open('{0}/{1}'.format(directory, file))
            w, h = im.size
            im.crop((0, 75, w, h)).save('{0}/rc_vocab_{1}_{2}.png'.format(destination_directory, num, ind))


if __name__ == '__main__':
    # slicer()
    # renamer()
    image_trim()
