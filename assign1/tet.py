from PIL import Image

PIL_Version = Image.VERSION

img_filename = '/Users/Vijay/Desktop/1.png'
im = Image.open(img_filename)
im.show()