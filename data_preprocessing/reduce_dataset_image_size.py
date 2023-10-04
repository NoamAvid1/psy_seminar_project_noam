from PIL import Image

image_path = "Hand_0000003.jpg"
new_path = "Hand_0000003_small.jpg"

img = Image.open(image_path)
print(img.size)

img2 = img.resize((600, 450), Image.LANCZOS)
print(img2.size)
img2.save(new_path, optimize=True, quality=95)
