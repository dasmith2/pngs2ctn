import Image

bytes = open("10_10.CTN", "r").read()
boilerplate = bytes[:8 * 3]
footer = bytes[-(8 * 4):]

im = Image.open("shotInTheDark.png")
width, height = im.size

out = open("shotInTheDark.CTN", "w")
out.write(boilerplate)

points = 0

for y in range(height):
  for x in range(width):
    if im.getpixel((x, y)) != (255, 255, 255):
      points += 1

print points

out.write(chr(points / 256))
out.write(chr(points % 256))

for i in range(3):
  out.write(chr(0))
out.write(chr(1)) # of frames
for i in range(2):
  out.write(chr(0))
  

for y in range(height):
  for x in range(width):
    if im.getpixel((x, y)) != (255, 255, 255):
      for d in range(3):
        # x
        out.write(chr((x + 128) % 256))
        out.write(chr(0))
        # y
        out.write(chr((127 - y + 256) % 256))
        out.write(chr(0))
        # color
        out.write(chr(0))
        out.write(chr(0))
        out.write(chr(0))
        out.write(chr(8))

out.write(footer)