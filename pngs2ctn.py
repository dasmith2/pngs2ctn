"""
This script takes a png file and turns it into one or more .CTN files. Any edge
between white and non-white in the .png will become a green laser line. This is
perfect for paint-by-laser, and not very helpful at all for creating laser
animations.

Therefore, TODO: Add a laser animation mode eventually. Probably never. It
should take a list of input files, each of which will become a frame in a
single output CTN file. It should interpret .pngs differently. I'm thinking
this mode interprets black as the background, and any border between black and
non-black becomes the non-black color.

TODO: Figure out how this projector does color, exactly.

python png2ctn.py -i ball.png
"""
from collections import namedtuple
from math import ceil
from optparse import OptionParser
# PIL! On one default install you're PIL, on another you're Image.
# Get it together man.
try:
  from PIL import Image, ImageDraw
except ImportError:
  import Image, ImageDraw
import random
from sets import Set
import sys


random.seed("I believe in a fing called love.")


MAX_STEP_SIZE = 5.0 / 500.0


# Shrink images down to this size first. That saves a lot of work.
MAX_IMAGE_SIZE = 1000


class CTNWriter:
  """ Here it is. You want to understand the CTN format? This is the class for
  you. The CTN file format is really pretty simple. It's an array of frames.
  Each frame has some boilerplate, then a few values describing how many points
  there are and which frame this is and out of how many. Then it lists all the
  points.  Each point is 2 bytes for x, 2 bytes for y, and 4 bytes for color. x
  goes up from left to right starting with 0x8000. In other words,
  1000000000000000 is on the far left, 0000000000000000 is halfway, and
  0111111111111111 is on the far right. y goes up from bottom to top starting
  with 0x8000. Adjacent points must be at most MAX_STEP_SIZE apart, otherwise
  the laser projector will truncate your design as soon as it hits a gap that's
  too big.

  Finally there's a little boilerplate at the bottom.

  For my purposes, each CTN is only ever going to have 1 frame. But seeing as
  this is the only documentation that exists on this file format, I'll write it
  all out. """
  # Why the R there? Beats me, team. Beats. Me.
  BOILER_PLATE = 'CRTN' + '\x00' * 20

  def write(self, ctn, output_file_path, repeat_points):
    out = open(output_file_path, "w")
    frame_count = self._2B_int(len(ctn.frames))
    for frame_index, frame in enumerate(ctn.frames):
      point_array = bytearray()
      point_count = 0
      last_point = None
      for path in frame.laserPaths:
        for point in path.get_points(repeat_points):
          # Reminder: a point is a namedtuple (x, y) where x and y are between
          # 0.0 and 1.0 inclusive.
          if point_to_point(last_point, point) > MAX_STEP_SIZE:
            raise Exception(
                "This design is invalid because it has 2 points that are too "
                "far apart. The projector will truncate it.")
          last_point = point
          point_count += 1
          point_array.extend(self._2B_position(point.x))
          # I'm used to y = 0 meaning the top so that's how I programmed it,
          # but .CTNs put y = 0 at the bottom.
          point_array.extend(self._2B_position(1.0 - point.y))
          # I don't understand how color works exactly. I didn't need to and I
          # got lazy.
          if path.laser_on():
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
    less clear. I "& 0x00FFFF" here because _2B_position causes an overflow bit
    half the time. """
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


Point = namedtuple('Point', ['x', 'y'])


class LaserPath:
  pass


class Connector(LaserPath):
  def __init__(self, from_point, to_point):
    self.from_point = from_point
    self.to_point = self.to_point

  def laser_on(self):
    return False

  def point_count(self, repeat_points=1):
    distance = Distance.point_2_point(self.from_point, self.to_point)
    return int(ceil((distance + 1e-5) / MAX_STEP_SIZE))

  def get_points(self, repeat_points=1):
    point_count = self.point_count(self.from_point, self.to_point)
    to_return = []
    # Don't bother repeating points where the laser is off. Accuracy only
    # matters when the laser is on.
    for point_i in range(point_count):
      frm_x = self.from_point.x
      frm_y = self.from_point.y
      to_return.append(Point(
          frm_x + 1.0 * point_i * (self.to_point.x - frm_x) / point_count,
          frm_y + 1.0 * point_i * (self.to_point.y - frm_y) / point_count))
    return to_return


class Feature(LaserPath):
  def __init__(self, points):
    self.points = points

  def laser_on(self):
    return True

  def point_count(self, repeat_points=1):
    return len(self.points) * repeat_points

  def rotate_to(self, starting_point):
    """ We have to rotate the points depending on where the connectors
    connect so the laser path remains valid. """
    index = self.points.index(starting_point)
    self.points = self.points[index:] + self.points[:index]

  def get_points(self, repeat_points=1):
    # chain(*izip(*[self.points for i in range(repeat_points)]))
    # would have worked too, but I restrained myself.
    if repeat_points == 1:
      return self.points
    else:
      new_points = []
      for point in self.points:
        for i in range(repeat_points):
          new_points.append(point)
      return new_points


class Go:
  directions = range(8)
  UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT = directions

  @classmethod
  def next(cls, at, direction):
    if direction == cls.UP:
      return Point(at.x, at.y - 1)
    if direction == cls.RIGHT:
      return Point(at.x + 1, at.y)
    if direction == cls.DOWN:
      return Point(at.x, at.y + 1)
    if direction == cls.LEFT:
      return Point(at.x - 1, at.y)

    if direction == cls.UP_RIGHT:
      return Point(at.x + 1, at.y - 1)
    if direction == cls.DOWN_RIGHT:
      return Point(at.x + 1, at.y + 1)
    if direction == cls.DOWN_LEFT:
      return Point(at.x - 1, at.y + 1)
    if direction == cls.UP_LEFT:
      return Point(at.x - 1, at.y - 1)

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


class PngFeatureFinder:
  def get_features(self, input_file):
    self.image = Image.open(input_file)
    self.image_width, self.image_height = self.image.size
    self.max_image_dimention = max(self.image.size)
    if self.max_image_dimention > MAX_IMAGE_SIZE:
      ratio = 1.0 * MAX_IMAGE_SIZE / self.max_image_dimention
      self.image_width = int(ratio * self.image_width)
      self.image_height = int(ratio * self.image_height)
      self.max_image_dimention = int(ratio * self.max_image_dimention)
      self.image = self.image.resize((self.image_width, self.image_height))
    return self._find_features()

  def _find_features(self):
    features = []
    seen_border = Set()
    last_position = None
    for y in range(self.image_height):
      for x in range(self.image_width):
        at = Point(x, y)
        if not at in seen_border and self._is_border(at):
          seen_pixels, draw_points = self._compute_feature(at)
          for pixel in seen_pixels:
            seen_border.add(pixel)
          features.append(Feature(draw_points))
    return features

  def _is_border(self, at):
    if self._is_white(at):
      return False
    for check in Go.adjacent_4(at):
      if not self._is_valid(check) or self._is_white(check):
        return True
    return False

  def _is_white(self, at):
    # A pixel can have 3 or 4 values depending on whether there's an alpha
    # chanel. 255 is pure white, but use 250 for a helpful fudge factor. The
    # Gimp or whatever is always leaving ALMOST white pixels all over the
    # place.
    return all([chanel > 250 for chanel in self.image.getpixel(at)])

  def _is_valid(self, at):
    return at.x >= 0 and at.x < self.image_width and at.y >= 0 \
           and at.y < self.image_height

  def _pixel_to_draw(self, at):
    return Point(
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
    draw_points = [] # The points the laser projector should draw.
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
      next_draw_point = self._pixel_to_draw(Point(draw_x, draw_y))
      # If we started on a pointy bit, when we come back around to the very
      # first pixel it'll get a second draw_point. That means we often will not
      # notice we're repeating unless we also pay attention to the second
      # pixel. Hence, this 2 here.
      if next_draw_point in draw_points[:2]:
        break
      draw_points.append(next_draw_point)

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
    return seen_pixels, draw_points


class Distance:
  @classmethod
  def point_2_point(cls, frm, to):
    """ TODO: I'm calculating the distance between a and b as cartesian, a.k.a.
    ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** .5. But x and y are controlled
    with completely separate mirrors. So as far as the machine is concerned,
    distance is probably more like min([abs(a.x - b.x), abs(a.y - b.y)]). Run a
    script over the sample CTNs and see what the maximum x delta, y delta, and
    cartesian delta are. """
    return ((to.x - frm.x) ** 2 + (to.y - frm.y) ** 2) ** .5

  @classmethod
  def point_2_feature(cls, point, feature):
    points = feature.get_points(1)
    if len(points) == 0:
      raise Exception("This feature doesn't have any points.")
    min_fpoint = None
    min_d = sys.maxint
    for fpoint in points:
      next_d = Distance.point_2_point(point, fpoint)
      if next_d < min_d:
        min_fpoint = fpoint
        min_d = next_d
    return min_fpoint, min_d

  @classmethod
  def feature_2_feature(cls, feature1, feature2):
    min_d = sys.maxint
    for f1point in feature1.get_points(1):
      f2point, next_min_d = Distance.point_2_feature(
          f1point, feature2)
      if next_min_d < min_d:
        min_f1point, min_f2point = f1point, f2point
        min_d = next_min_d
    return min_f1point, min_f2point, min_d


class CTNCreator:
  """ Here's where we take our beautiful abstract list of features and turn
  them into gritty CTN files that work OK in real life. """
  def get_CTNs(self, features, repeat_points, max_points):
    for index in range(len(features)):
      features[index] = self._spread_out_draw_points(features[index])

    big_ctns = []
    for feature in features:
      if feature.point_count(repeat_points) > max_points:
        big_ctns.append(self._ctn_from_paths([feature]))
        features.remove(feature)

    list_count = 1
    try_with_each_count = 3
    min_path_lists = None
    min_worst_distance = sys.maxint
    while not min_path_lists:
      for i in range(try_with_each_count):
        path_lists = self._greedy_path_list_builder(
            features, list_count, repeat_points, max_points)
        if path_lists:
          worst_distance = self._worst_distance(path_lists)
          if min_path_lists == None or worst_distance < min_worst_distance:
            min_worst_distance = worst_distance
            min_path_lists = path_lists
      list_count += 1

    ctns = [self._ctn_from_paths(pl.get_paths()) for pl in min_path_lists]
    return big_ctns + ctns

  def _ctn_from_paths(self, paths):
    return CTN([CTNFrame(paths)])

  class _PathList:
    def __init__(self, first_feature, repeat_points):
      self.features = [first_feature]
      self.point_count = feature.point_count(repeat_points)
      self.last_point = None
      # Are there ANY features left that won't put us over the limit?
      self.some_left = True

    def add_feature(self, connector, feature, repeat_points):
      self.connectors.append(connector)
      self.features.append(feature)
      self.point_count += connector.point_count(repeat_points) + \
                          feature.point_count(repeat_points)
      self.last_point = connector.to_point

    def get_paths(self):
      if len(self.features) == 1:
        return self.features
      paths = []
      for index, connector in enumerate(self.connectors):
        feature_before = self.features[index]
        feature_before.rotate_to(connector.from_point)
        paths.append(feature_before)
        paths.append(connector)
      last_feature = self.features[-1]
      last_connector = self.connectors[-1]
      last_feature.rotate_to(last_connector.to_point)
      paths.append(last_feature)
      return paths

  def _greedy_path_list_builder(
      self, features, list_count, repeat_points, max_points):
    """ Make list_count _PathLists, each started with a random feature.
    Take the _PathList with the least points, a.k.a. the shortest, and
    add to it the next feature with the shortest gap whose addition won't put
    the _PathList over the limit. In other words I'm trying to grow the
    _PathLists as evenly as I can while also trying to minimize the time
    the laser spends between features. It's a traveling salesmen problem with
    multiple salesmen 'solved' with a greedy algorithm. """
    features_copy = list(features)
    path_lists = []
    for i in range(list_count):
      random_feature = random.choice(features_copy)
      features_copy.remove(random_feature)
      path_lists.append(self._PathList(random_feature, repeat_points))

    while len(features_copy) > 0:
      available = [f for f in path_lists if f.some_left]
      if not available:
        return None
      available.sort(lambda x, y: x.point_count - y.point_count)

      shortest = available[0]
      min_next_feature = None
      min_next_d = sys.maxint
      min_next_connection_point
      for feature2 in features_copy:
        if shortest.last_point:
          last_feature_point = shortest.last_point
          next_feature_point, next_d = Distance.point_2_feature(
              last_feature_point, feature2)
        else:
          last_feature_point, next_feature_point, next_d = \
              Distance.feature_2_feature(shortest.features[-1], feature2)
        connector = Connector(last_feature_point, next_feature_point)
        next_d = connector.point_count(repeat_points)
        more_points = next_d + feature2.point_count(repeat_points)
        under_max = shortest.point_count + more_points <= max_points
        if next_d < min_next_d and under_max:
          min_next_d = next_d
          min_next_connector = connector
          min_next_feature = feature2
      if min_next_feature:
        shortest.add_feature(min_connector, min_next_feature, repeat_points)
        feature_copy.remove(min_next_feature)
      else:
        shortest.some_left = False

  def _worst_distance(self, path_lists):
    return max([fl.point_count for fl in path_lists])

  def _spread_out_draw_points(self, feature):
    i = 0
    draw_points = feature.get_points(1)
    return_draw_points = []
    while i < len(draw_points):
      frm = draw_points[i]
      return_draw_points.append(frm)
      to_offset = 1
      to = draw_points[(i + to_offset) % len(draw_points)]
      d = Distance.point_2_point(frm, to)
      if d >= MAX_STEP_SIZE:
        if d > MAX_STEP_SIZE:
          return_draw_points.extend(self._points_from_to(frm, to))
        i += 1
      else:
        while d < MAX_STEP_SIZE:
          at_the_end = i + to_offset == len(draw_points)
          if at_the_end:
            # We need to connect the circle, so to speak. The laser has to be
            # able to trace all the way around this feature because we may
            # choose to start drawing this feature anywhere in the middle.
            if Distance.point_2_point(frm, draw_points[0]) > MAX_STEP_SIZE:
              # a.k.a. i + to_offset - 1, a.k.a. len(draw_points) - 1
              return_draw_points.append(draw_points[-1])
            return Feature(return_draw_points)
          to_offset += 1
          if i + to_offset < len(draw_points):
            d = Distance.point_2_point(frm, draw_points[i + to_offset])
        i = i + to_offset - 1 # Went too far, so back up 1.
    return Feature(return_draw_points)


CTN = namedtuple('CTN', ['frames'])


CTNFrame = namedtuple('CTNFrame', ['paths'])


class FrameDebuger:
  def draw_debug(self, frame, output_file_name):
    debug_size = 750
    debug = Image.new("1", (debug_size, debug_size))
    draw = ImageDraw.Draw(debug)
    for path in frame.paths:
      points = [self._draw_2_debug(c, debug_size) for c in path.get_points(1)]
      for index, from_point in enumerate(points):
        to_point = points[(index + 1) % len(points)]
        fx = from_point.x
        fy = from_point.y
        if path.laser_on():
          draw.line((fx, fy, to_point.x, to_point.y), fill=1)
        draw.line(self._valid_line(fx - 1, fy, fx + 1, fy, debug_size), fill=1)
        draw.line(self._valid_line(fx, fy - 1, fx, fy + 1, debug_size), fill=1)
    debug.save(open(output_file_name, "w"), "PNG")

  def _draw_2_debug(self, point, dimension):
    return Point(int(point.x * dimension), int(point.y * dimension))

  def _valid_line(self, x1, y1, x2, y2, maximum_exclusive):
    return self._closest_valid(x1, y1, maximum_exclusive) + \
           self._closest_valid(x2, y2, maximum_exclusive)

  def _closest_valid(self, x, y, maximum_exclusive):
    maximum = maximum_exclusive - 1
    return (min(maximum, max(0, x)), min(maximum, max(0, y)))


def main():
  parser = OptionParser()
  parser.add_option(
      "-i", "--input", dest="input_file",
      help=("A png file. I'll draw a laser path along each border between "
            "white and nonwhite. So for example, a solid black disk will "
            "become a circle around the circumference of the disk."))
  parser.add_option(
      "-o", "--output_file_prefix", dest="output_file_prefix",
      default="output",
      help="Each output CTN file will get this prefix.")
  parser.add_option(
      "-d", "--debug", action="store_true", dest="debug", default=False,
      help="Output a debug .png file for every generated CTN file.")
  parser.add_option(
      "-r", "--repeat_points", dest="repeat_points", default=3,
      help=("How long should the laser linger as it draws? Higher values "
            "mean more accuracy but more flickering from a slower laser. "
            "Technically, this value says how many times to repeat each point "
            "when the laser is on."))
  parser.add_option(
      "-m", "--max_points", dest="max_points", default=1000,
      help=("A CTN file is a list of points. The laser projector attempts to "
            "point the laser at each point in succession. Use this option to "
            "limit how many points may be in a CTN file before we start "
            "splitting the result across multiple files. This is essential for "
            "complex designs with many points. This blue laser projector can "
            "only handle so many points before it flickers so much it's "
            "unusable."))
  (options, args) = parser.parse_args()

  input_file = options.input_file
  if not input_file.lower().endswith(".png"):
    raise Exception("This script only accepts .png files as input.")

  features = PngFeatureFinder().get_features(input_file)
  ctns = CTNCreator().get_CTNs(
      features, options.repeat_points, options.max_points)
  for index, ctn in enumerate(ctns):
    CTNWriter().write(ctn, "%s_%s.CTN" % (options.output_file_prefix, index + 1))
    if options.debug:
      # Again, for my purposes, there's only ever going to be one frame.
      for frame_index, frame in enumerate(ctn.frames):
        args = (options.output_prefix, index + 1, frame_index + 1)
        FrameDebugger().draw_debug(frame, "%s_%s_%s_debug.png" % args)


if __name__ == "__main__":
  main()
