"""
The CTN file format is really pretty simple. It starts with 16 bytes of
boilerplate. Then you've got an array of frames. Each frame has a 16
byte header which includes info on how many points are in the frame.
Then it's a list of points. Each point is 2 bytes for x, 2 bytes for y,
and 4 bytes for color. x and y are ints where (0, 0) is the center, not in a
corner like you might expect. y increases as you move up and x increases as
you move right. The minimum x or y is -32768 and the maximum is 32768. So for
example the point (-32768, -32768) is in the lower left corner.

python pngs2ctn.py -i ball_up.png,ball_middle.png,ball_down.png,ball_middle.png -o bouncing_ball.CTN
"""
from collections import namedtuple
from optparse import OptionParser
from sets import Set
from PIL import Image

class CTN:
  def __init__(self):
    self.frames = []

  def add_frame(self, ctn_frame):
    frames.append(ctn_frame)

  def write(self, output_file):
    out = open(output_file, "w")
    out.write(self.header_boilerplate())
    for frame in self.frames()
      out.write(frame.get_bytes())
    out.write(self.footer_boilerplate())

  def header_boilerplate(self):
    header = bytearray()
    header.extend((0,) * 8)
    return header

  def footer_boilerplate(self):
    footer = bytearray()
    footer.extend((0,) * 8)
    return footer


Point = namedtuple('Point', ['x', 'y', 'color'])

Coordinate = namedtuple('Coordinate', ['x', 'y'])

directions = range(8)

Class Go:
  UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT = directions

  @classmethod
  def next(cls, at, direction):
    if direction == cls.UP:
      return Coordinate(at.x, at.y - 1)
    if direction == cls.RIGHT:
      return Coordinate(at.x + 1, at.y)
    if direction == cls.DOWN:
      return Coordinate(at.x, at.y + 1)
    if direction == cls.LEFT:
      return Coordinate(at.x - 1, at.y)

    if direction == cls.UP_RIGHT:
      return Coordinate(at.x + 1, at.y - 1)
    if direction == cls.DOWN_RIGHT:
      return Coordinate(at.x + 1, at.y + 1)
    if direction == cls.DOWN_LEFT:
      return Coordinate(at.x - 1, at.y + 1)
    if direction == cls.UP_LEFT:
      return Coordiante(at.x - 1, at.y - 1)

  @classmethod
  def direction_from_to(cls, frm, to):
    if to.x > frm.x:
      if to.y > frm.y:
        return cls.DOWN_RIGHT
      elif to.y == frm.y:
        return cls.RIGHT
      elif to.y < frm.y:
        return cls.UP_RIGHT
    elif to.x < frm.x:
      if to.y > frm.y:
        return cls.DOWN_LEFT
      elif to.y == frm.y:
        return cls.LEFT
      elif to.y < frm.y:
        return cls.UP_LEFT
    elif to.x == frm.x:
      if to.y > frm.y:
        return cls.DOWN
      elif to.y < frm.y:
        return cls.UP

  @classmethod
  def adjacent_4(cls, at):
    return [cls.next(at, cls.UP), cls.next(at, cls.RIGHT), cls.next(at, cls.DOWN), cls.next(at, cls.LEFT)]

  @classmethod
  def adjacent_8(cls, at):
    return [
        cls.next(at, cls.UP), cls.next(at, cls.UP_RIGHT),
        cls.next(at, cls.RIGHT), cls.next(at, cls.DOWN_RIGHT),
        cls.next(at, cls.DOWN), cls.next(at, cls.DOWN_LEFT),
        cls.next(at, cls.LEFT), cls.next(at, cls.UP_LEFT)]


Feature = namedtuple('Feature', ['pixels', 'draw_points'])


class CTNFrame:
  def __init__(self, override_color=None):
    self.points = ()
    self.override_color = override_color
    self.features = set()

  def write_debug(self, file_name):
    debug = Image.new("w", (500, 500))
    for feature in self.features:
      for (index, point) in enumerate(feature.draw_points):
        to_point = feature.draw_points[(index + 1)]

  def load_from_png(self, input_file):
    image = Image(input_file)
    self.image_width, self.image_height = im.size
    self.find_features()

  def find_features(self):
    seen_border = Set()
    for y in range(self.image_height):
      for x in range(self.image_width):
        at = Coordinate(x, y)
        if not at in seen_border and self.is_border(at):
          feature = self.compute_feature(at)
          for pixel : feature.pixels:
            seen_border.add(pixel)
          self.features.add(feature)

  def is_border(self, at):
    if self.is_white(at):
      return False
    for check in Go.adjacent_4(at)
      if not self.is_valid_coord(check) or self.is_white(check):
        return True
    return False

  def is_valid_coord(self, at):
    return at.x >= 0 and at.x < self.image_width and at.y >= 0 and at.y < self.image_height

  def compute_feature(self, at):
    """ Walk around the outside of a non-white blob. Keep adding points until
    we hit a spot we added already. """
    for index, go in directions:
      looking_at = Go.next(at, go)
      if not self.is_valid(looking_at) or self.is_white(looking_at):
        outward = index
        break
    draw_coords = []
    seen_pixels = set()
    while not at in seen_pixels:
      seen_pixels.add(at)

      draw_x, draw_y = at.x, at.y
      if outward in (Go.DOWN_RIGHT, Go.DOWN, Go.DOWN_LEFT):
        draw_y += 1
      if outward in (Go.UP_RIGHT, Go.RIGHT, Go.DOWN_RIGHT):
        draw_x += 1
      draw_coords.append(Coordinate(draw_x, draw_y))

      for go_offset in range(len(directions)):
        looking_around = (outward + go_offset) % len(directions)
        looking_at = Go.next(at, looking_around)
        if self.is_valid(looking_at) and not self.is_border(looking_at):
          outward = (Go.direction_from_to(looking_at, at) + 1) % len(directions)
          at = looking_at
          break
    return Feature(seen_pixels, draw_coords)

  def add_point(self, image, at):
    self.points.append(Point(at.x, at.y, self.override_color or image.get((x, y))))

  def get_bytes(self):
    bytes = bytearray()
    self.append_header(bytes)
    self.append_body(bytes)
    return bytes

  def append_header(self, bytes):
    pass

  def append_body(self, bytes):
    for point in points:
      append_double_byte(bytes, point.x)
      append_double_byte(bytes, point.y)
      append_double_byte(0)
      append_double_byte(bytes, point.color)

  def append_double_byte(self, bytes, my_int):
    bytes.append(my_int / 256)
    bytes.append(my_int % 256)


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input_files", help="A comma delimited list of png files, each of which will become a single frame in the ctn")
  parser.add_option("-o", "--output", dest="output_file", help="The output .CTN file.")
  parser.add_option("-d", "--debug", action="store_true", dest="debug", default=False, help="Output a debug .png file for every input file.")
  (options, args) = parser.parse_args()
  output = options.output_file
  if not output.lower().endswith(".ctn"):
    output += ".CTN"

  ctn = CTN()

  for input_file in options.input_files.split(","):
    if not input_file.lower().endswith(".png"):
      raise Exception("This script only accepts .png files as input.")
    ctn_frame = CTNFrame()
    ctn_frame.load_from_png(input_file)
    ctn.add_frame(ctn_frame)
    if options.debug:
      ctn_frame.write_debug(input_file[:-4] + "_debug.png")

  ctn.write(output)

