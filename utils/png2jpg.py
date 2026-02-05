from PIL import Image

PATH = './source/hand_writing.png'
img = Image.open(PATH)
img = img.convert("RGB")
img.save("hand_writing.jpg", "JPEG")