"""
The CTN file format is really pretty simple. It's an array of frames. Each
frame has some boilerplate, then a few values describing how many points there
are and which frame this is and out of how many. Then it lists all the points.
Each point is 2 bytes for x, 2 bytes for y, and 4 bytes for color. x goes up
from left to right starting with 0x8000. In other words, 1000000000000000 is on
the far left, 0000000000000000 is halfway, and 0111111111111111 is on the far
right. y goes up from bottom to top starting with 0x8000. Each point must be
at least MAX_STEP_SIZE close to each other, otherwise the laser projector will
truncate your design as soon as it hits a gap that's too big.

Finally there's a little boilerplate at the bottom.

This script takes a list of png file names and turns those into the frames of
a single .CTN file. Any edge between white and non-white in the .png will
become a green laser line. This is perfect for paint-by-laser, and not very
helpful at all for creating laser animations.

Therefore, TODO: Add a laser animation mode which interprets .pngs differently.
I'm thinking this mode interprets black as the background, and any border
between black and non-black becomes the non-black color.

Also, TODO: Figure out how color works, exactly.

python pngs2ctn.py -i ball_up.png,ball_middle.png,ball_down.png,ball_middle.png -o bouncing_ball.CTN
"""
from collections import namedtuple
from math import ceil
from optparse import OptionParser
from PIL import Image, ImageDraw
import random
from sets import Set
import sys


random.seed("There's no love like your love.")


""" TODO: I'm calculating the distance between a and b as cartesian, a.k.a.
((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** .5. But x and y are controlled with
completely separate mirrors. So as far as the machine is concerned, distance
is probably more like min([abs(a.x - b.x), abs(a.y - b.y)]). Run a script over
the sample CTNs and see what the maximum x delta, y delta, and cartesian delta
are. """
MAX_STEP_SIZE = 5.0 / 500.0


Coord = namedtuple('Coord', ['x', 'y'])


class CTN:
  BOILER_PLATE = 'CRTN' + '\x00' * 20

  def __init__(self):
    self.frames = []

  def add_frame(self, ctn_frame):
    self.frames.append(ctn_frame)

  def write(self, output_file_path):
    out = open(output_file_path, "w")
    frame_count = self._2B_int(len(self.frames))
    for frame_index, frame in enumerate(self.frames):
      point_array = bytearray()
      point_count = 0
      for feature in frame.features:
        for point in feature.points:
          point_count += 1
          point_array.extend(self._2B_position(point.x))
          # I'm used to y = 0 meaning the top so that's how I programmed it,
          # but .CTNs put y = 0 at the bottom.
          point_array.extend(self._2B_position(1.0 - point.y))
          if feature.color > 0:
            point_array.extend('\x00\x00\x00\x08')
          else:
            point_array.extend('\x00\x00\x40\x00')
      self._write_frame_header(out, point_count, frame_index, frame_count)
      out.write(point_array)
    self._write_footer(out, frame_count)

  def _2B_position(self, value):
    """ value is in the range [0.0, 1.0]. The maximum unsigned int you can
    express with 2 bytes is 0xFFFF. So self._2B_int(int(value * 0xFFFF)) is
    what we'd return if 0x0 meant 0 in .CTN files. However, 0x8000 means 0. So
    we add our result to 0x8000. """
    return self._2B_int(0x8000 + int(value * 0xFFFF))

  def _2B_int(self, value):
    """ struct.pack('>I', value)[-2:] works too but it's slightly slower and
    less clear. """
    return chr((value & 0x00FFFF) / 256) + chr(value % 256)

  def _write_frame_header(self, out, point_count, frame_index, frame_count):
    out.write(self.BOILER_PLATE)
    out.write(self._2B_int(point_count))
    out.write(self._2B_int(frame_index))
    out.write(frame_count)
    out.write('\x00' * 2)

  def _write_footer(self, out, frame_count):
    out.write(self.BOILER_PLATE)
    out.write('\x00' * 4)
    out.write(frame_count)
    out.write('\x00' * 2)




class Go:
  directions = range(8)
  UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT = directions

  @classmethod
  def next(cls, at, direction):
    if direction == cls.UP:
      return Coord(at.x, at.y - 1)
    if direction == cls.RIGHT:
      return Coord(at.x + 1, at.y)
    if direction == cls.DOWN:
      return Coord(at.x, at.y + 1)
    if direction == cls.LEFT:
      return Coord(at.x - 1, at.y)

    if direction == cls.UP_RIGHT:
      return Coord(at.x + 1, at.y - 1)
    if direction == cls.DOWN_RIGHT:
      return Coord(at.x + 1, at.y + 1)
    if direction == cls.DOWN_LEFT:
      return Coord(at.x - 1, at.y + 1)
    if direction == cls.UP_LEFT:
      return Coord(at.x - 1, at.y - 1)

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
    return [
        cls.next(at, cls.UP), cls.next(at, cls.RIGHT),
        cls.next(at, cls.DOWN), cls.next(at, cls.LEFT)]


Feature = namedtuple('Feature', ['color', 'points'])


class CTNFrame:
  def __init__(self):
    self.features = []

  def write_debug(self, file_name):
    debug_size = 750
    debug = Image.new("1", (debug_size, debug_size))
    draw = ImageDraw.Draw(debug)
    for feature in self.features:
      points = [self._draw_2_debug(c, debug_size) for c in feature.points]
      for index, from_point in enumerate(points):
        to_point = points[(index + 1) % len(points)]
        fx = from_point.x
        fy = from_point.y
        if feature.color > 0:
          draw.line((fx, fy, to_point.x, to_point.y), fill=1)
        draw.line(self._valid_line(fx - 1, fy, fx + 1, fy, debug_size), fill=1)
        draw.line(self._valid_line(fx, fy - 1, fx, fy + 1, debug_size), fill=1)
    debug.save(open(file_name, "w"), "PNG")

  def _valid_line(self, x1, y1, x2, y2, maximum_exclusive):
    return self._closest_valid(x1, y1, maximum_exclusive) + self._closest_valid(x2, y2, maximum_exclusive)

  def _closest_valid(self, x, y, maximum_exclusive):
    # Turns -1 into 0 and 500 into 499, but leaves 10 as 10.
    maximum = maximum_exclusive - 1
    return (min(maximum, max(0, x)), min(maximum, max(0, y)))

  def _draw_2_debug(self, coord, dimension):
    return Coord(int(coord.x * dimension), int(coord.y * dimension))

  def load_from_png(self, input_file):
    self._load_png(input_file)
    self._find_features()
    self._sort_and_connect_features()

  def _load_png(self, input_file):
    self.image = Image.open(input_file)
    self.image_width, self.image_height = self.image.size
    self.max_image_dimention = max(self.image.size)

  def _find_features(self):
    seen_border = Set()
    last_position = None
    for y in range(self.image_height):
      for x in range(self.image_width):
        at = Coord(x, y)
        if not at in seen_border and self._is_border(at):
          seen_pixels, draw_coords = self._compute_feature(at)
          for pixel in seen_pixels:
            seen_border.add(pixel)
          self.features.append(Feature(255, draw_coords))

  def _sane_feature_shuffle(self, feature_copy):
    feature_copy = list(feature_copy)
    on = feature_copy[random.randint(0, len(feature_copy) - 1)]
    shuffled = []
    last_point = None

    shuffled.append(on)
    feature_copy.remove(on)

    while len(feature_copy) > 0:
      min_next_feature = None
      min_next_d = sys.maxint
      min_next_feature_point = None
      for feature2 in feature_copy:
        if last_point:
          next_feature_point, next_d = self._point_to_feature(
              last_point, feature2)
        else:
          throw_away, next_feature_point, next_d = \
              self._feature_to_feature_dist(on, feature2)
        if next_d < min_next_d:
          min_next_feature = feature2
          min_next_d = next_d
          min_next_feature_point = next_feature_point
      on = min_next_feature
      shuffled.append(on)
      feature_copy.remove(on)
    return shuffled

  def _sort_and_connect_features(self):
    """ You want to minimize the time moving the laser between features.
    This is basically traveling salesman, so just poke around for a while
    and take the best path we can find. Start with a sane feature ordering
    where you pick one at random, and greedily add the closest feature one at
    a time. Then do a few halfassed bubble sort moves. Repeat all that a few
    times and keep the best one. I arrived at this algorithm by dicking around
    on one fairly complicated design. Good enough for art! There's room for
    improvement though. Going from features A -> B -> C means going from points
    a1 -> b1 -> c1. Right now b1 is greedily optimized based solely on the
    shortest pair of points between A and B, so b1 doesn't take feature C into
    account at all. """
    best_d = sys.maxint
    best_connectors = None
    best_features = self.features
    feature_count = len(self.features)
    for start_fresh in range(5):
      feature_copy = self._sane_feature_shuffle(self.features)
      current_d, current_connectors = self._total_laser_gap_dist(feature_copy)
      for delta in range(1, 5):
        for i in range(feature_count - delta):
          self._swap_features(feature_copy, i, i + delta)
          new_d, new_connectors = self._total_laser_gap_dist(feature_copy)
          if new_d < current_d:
            making_progress = True
            current_d = new_d
            current_connectors = new_connectors
          else:
            self._swap_features(feature_copy, i, i + delta)
      if current_d < best_d:
        best_d = current_d
        best_connectors = current_connectors
        best_features = list(feature_copy)
    self.features = best_features
    self.features = self._features_with_connectors(best_connectors)

  def _swap_features(self, features, frm, to):
    t = features[frm]
    features[frm] = features[to]
    features[to] = t

  def _features_with_connectors(self, connectors):
    features_with_connectors = []
    if len(connectors) == 0:
      if len(self.features) == 1:
        return [self.features[0]]
      else:
        raise Exception("I got no connectors, but I have more than 1 feature.")
    for i, feature in enumerate(self.features):
      connector = connectors[i]
      i_offset = feature.points.index(connector[0])
      new_feature_points = feature.points[i_offset:] + feature.points[:i_offset]
      connector_points = self._points_from_to(connector[0], connector[1])
      features_with_connectors += [Feature(feature.color, new_feature_points), Feature(0, connector_points)]
    return features_with_connectors

  def _total_laser_gap_dist(self, feature_list):
    """ Use this to help us sort the features so the laser has to travel a
    reasonably not-stupid distance. """
    if len(feature_list) == 0:
      raise Exception("Extected a non-zero feature list")
    if len(feature_list) == 1:
      return 0.0, []
    min_f1point, min_f2point, min_d = self._feature_to_feature_dist(
        feature_list[0], feature_list[1])
    total = min_d
    connectors = [(min_f1point, min_f2point)]
    at = min_f2point
    if len(feature_list) > 2:
      for feature_n in feature_list[2:]:
        f_n_min_point, next_min_d = self._point_to_feature(
            at, feature_n)
        total += next_min_d
        connectors.append((at, f_n_min_point))
        at = f_n_min_point
    total += self._point_to_point(at, min_f1point)
    connectors.append((at, min_f1point))
    return total, connectors

  def _feature_to_feature_dist(self, feature1, feature2):
    min_d = sys.maxint
    for f1point in feature1.points:
      f2point, next_min_d = self._point_to_feature(
          f1point, feature2)
      if next_min_d < min_d:
        min_f1point, min_f2point = f1point, f2point
        min_d = next_min_d
    return min_f1point, min_f2point, min_d

  def _point_to_feature(self, point, feature):
    if len(feature.points) == 0:
      raise Exception("This feature doesn't have any points.")
    min_fpoint = None
    min_d = sys.maxint
    for fpoint in feature.points:
      next_d = self._point_to_point(point, fpoint)
      if next_d < min_d:
        min_fpoint = fpoint
        min_d = next_d
    return min_fpoint, min_d

  def _is_border(self, at):
    if self._is_white(at):
      return False
    for check in Go.adjacent_4(at):
      if not self._is_valid(check) or self._is_white(check):
        return True
    return False

  def _is_white(self, at):
    # A pixel can have 3 or 4 values depending on whether there's an alpha
    # chanel.
    return all([chanel == 255 for chanel in self.image.getpixel(at)])

  def _is_valid(self, at):
    return at.x >= 0 and at.x < self.image_width and at.y >= 0 \
           and at.y < self.image_height

  def _pixel_to_draw(self, at):
    return Coord(
        at.x * 1.0 / self.max_image_dimention,
        at.y * 1.0 / self.max_image_dimention)

  def _compute_feature(self, at):
    """ Walk around the outside of a non-white blob. Pretend you have your back
    to the feature, and outward keeps track of which direction your face is
    pointed. Keep moving to your right, adding points until you go all the way
    around to where you're about to draw the same point again. """
    for go in Go.directions:
      looking_at = Go.next(at, go)
      if not self._is_valid(looking_at) or self._is_white(looking_at):
        outward = go
        break
    draw_coords = [] # The coords the laser projector should draw.
    seen_pixels = set() # The actual pixels on the outside of this feature.
    while True:
      seen_pixels.add(at)

      # If we're on the bottom of a feature, you don't want to draw a point at
      # the top of the pixel.
      draw_x, draw_y = at.x, at.y
      if outward in (Go.DOWN_RIGHT, Go.DOWN, Go.DOWN_LEFT):
        draw_y += 1
      if outward in (Go.UP_RIGHT, Go.RIGHT, Go.DOWN_RIGHT):
        draw_x += 1
      next_draw_coord = self._pixel_to_draw(Coord(draw_x, draw_y))
      # If we started on a pointy bit, when we come back around to the very
      # first pixel it'll get a second draw_coord. Hence, we often will not
      # notice we're repeating unless we also pay attention to the second
      # pixel. Hence, this 2 here.
      if next_draw_coord in draw_coords[:2]:
        break
      draw_coords.append(next_draw_coord)

      # We're looking outward. Rotate to the right until we see the next
      # non-white pixel. That will be the next one to our right.
      for go_offset in range(len(Go.directions)):
        looking_around = (outward + go_offset) % len(Go.directions)
        looking_at = Go.next(at, looking_around)
        if self._is_valid(looking_at) and not self._is_white(looking_at):
          # To compute the new outward, look back where we came from, and
          # rotate once to the right.
          looking_back = Go.direction_from_to(looking_at, at)
          outward = (looking_back + 1) % len(Go.directions)
          at = looking_at
          break
    return seen_pixels, self._spread_out_draw_coords(draw_coords)

  def _spread_out_draw_coords(self, draw_coords):
    i = 0
    return_draw_coords = []
    while i < len(draw_coords):
      frm = draw_coords[i]
      return_draw_coords.append(frm)
      to_offset = 1
      to = draw_coords[(i + to_offset) % len(draw_coords)]
      d = self._point_to_point(frm, to)
      if d >= MAX_STEP_SIZE:
        if d > MAX_STEP_SIZE:
          return_draw_coords.extend(self._points_from_to(frm, to))
        i += 1
      else:
        while d < MAX_STEP_SIZE:
          if i + to_offset == len(draw_coords):
            # We need to connect the circle, so to speak. The laser has to be
            # able to trace all the way around this feature because we may
            # choose to start drawing this feature anywhere in the middle.
            if self._point_to_point(frm, draw_coords[0]) > MAX_STEP_SIZE:
              # a.k.a. i + to_offset - 1, a.k.a. len(draw_coords) - 1
              return_draw_coords.append(draw_coords[-1])
            return return_draw_coords
          to_offset += 1
          if i + to_offset < len(draw_coords):
            d = self._point_to_point(frm, draw_coords[i + to_offset])
        i = i + to_offset - 1 # Went too far, so back up 1.
    return return_draw_coords

  def _points_from_to(self, frm, to):
    d = self._point_to_point(frm, to)
    if d <= MAX_STEP_SIZE:
      return [frm, to]
    step_count = int(ceil(d / MAX_STEP_SIZE))
    to_return = []
    for step_i in range(self._to_int(step_count)):
      to_return.append(Coord(
          frm.x + 1.0 * step_i * (to.x - frm.x) / step_count,
          frm.y + 1.0 * step_i * (to.y - frm.y) / step_count))
    last_d = self._point_to_point(to, to_return[-1])
    return to_return

  def _to_int(self, flt):
    # Hacky. Whatever.
    return int(flt + 1e-5)

  def _point_to_point(self, frm, to):
    # See? Basic high school math. That shit is important!
    return ((to.x - frm.x) ** 2 + (to.y - frm.y) ** 2) ** .5

  def get_bytes(self):
    bytes = bytearray()
    self._append_header(bytes)
    self._append_body(bytes)
    return bytes

  def _append_header(self, bytes):
    pass

  def _append_body(self, bytes):
    for point in points:
      self._append_double_byte(bytes, point.x)
      self._append_double_byte(bytes, point.y)
      self._append_double_byte(0)
      self._append_double_byte(bytes, point.color)

  def _append_double_byte(self, bytes, my_int):
    bytes.append(my_int / 256)
    bytes.append(my_int % 256)


def main():
  parser = OptionParser()
  parser.add_option(
      "-i", "--input", dest="input_files",
      help=("A comma delimited list of png files, each of which will become a"
            "single frame in the ctn"))
  parser.add_option(
      "-o", "--output", dest="output_file",
      help="The output .CTN file.")
  parser.add_option(
      "-d", "--debug", action="store_true", dest="debug", default=False,
      help="Output a debug .png file for every input file.")
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


if __name__ == "__main__":
  main()
