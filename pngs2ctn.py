"""
<<<<<<< HEAD
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

Therefore, just in case anybody wants to make animations with this thing,
TODO: Add a laser animation mode which interprets .pngs differently.
I'm thinking this mode interprets black as the background, and any border
between black and non-black becomes the non-black color.
=======
TODO: Get consistent about points vs coords.

If you're here because you want to understand the CTN format for the blue
unbranded programmable laser projector from China, skip down to the CTNWriter
class. That's got everything you need to know.

This script takes a png file and turns it into one or more .CTN files. Any edge
between white and non-white in the .png will become a green laser line. This is
perfect for paint-by-laser, and not very helpful at all for creating laser
animations.

Therefore, TODO: Add a laser animation mode eventually. Probably never. It
should take a list of input files, each of which will become a frame in a
single output CTN file. It should interpret .pngs differently. I'm thinking
this mode interprets black as the background, and any border between black and
non-black becomes the non-black color.
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34

Also, TODO: Figure out how color works, exactly.

python pngs2ctn.py -i ball.png
"""
from collections import namedtuple
from math import ceil
from optparse import OptionParser
from PIL import Image, ImageDraw
import random
from sets import Set
import sys


random.seed("I believe in a fing called love.")


""" TODO: I'm calculating the distance between a and b as cartesian, a.k.a.
((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** .5. But x and y are controlled with
completely separate mirrors. So as far as the machine is concerned, distance
is probably more like min([abs(a.x - b.x), abs(a.y - b.y)]). Run a script over
the sample CTNs and see what the maximum x delta, y delta, and cartesian delta
are. """
MAX_STEP_SIZE = 5.0 / 500.0


""" If an image comes in with a height or width greater than this in pixels,
resize it before we process it. Close enough for rock and roll. """
MAX_IMAGE_DIMENSION = 1000


Coord = namedtuple('Coord', ['x', 'y'])


<<<<<<< HEAD
class CNTList:
  """ A list of CTN files to support the max_points option. """
  def __init__(self, max_points, input_file):
    next_ctn_count = 1
    last_points = sys.maxint
    ctn_frame = CTNFrame()
    ctn_frame.load_features(input_file)
    while last_points > max_points:
      frames = [CTNFrame() for i in range(next_ctn_count)]


class CTN:
  BOILER_PLATE = 'CRTN' + '\x00' * 20
=======
class CTNWriter:
  """ Here it is. You want to understand the CTN format? This is the class for
  you. The CTN file format is really pretty simple. It's an array of frames.
  Each frame has some boilerplate, then a few values describing how many points
  there are and which frame this is and out of how many. Then it lists all the
  points.  Each point is 2 bytes for x, 2 bytes for y, and 4 bytes for color. x
  goes up from left to right starting with 0x8000. In other words,
  1000000000000000 is on the far left, 0000000000000000 is halfway, and
  0111111111111111 is on the far right. y goes up from bottom to top starting
  with 0x8000. Each point must be at least MAX_STEP_SIZE close to each other,
  otherwise the laser projector will truncate your design as soon as it hits a
  gap that's too big.
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34

  Finally there's a little boilerplate at the bottom.

  For my purposes, each CTN is only ever going to have 1 frame. But seeing as
  this is the only documentation that exists on this file format, I'll write it
  all out. """
  BOILER_PLATE = 'CRTN' + '\x00' * 20 # Why cRtn? Beats me, team. Beats. Me.

  def __init__(self, ctn):
    self.ctn = ctn

  def write(self, output_file_path):
    out = open(output_file_path, "w")
    frame_count = self._2B_int(len(self.ctn.frames))
    for frame_index, frame in enumerate(self.ctn.frames):
      point_array = bytearray()
      point_count = 0
      last_point = None
      for feature in frame.features:
        for point in feature.points:
<<<<<<< HEAD
          yes_color = feature.color > 0
          # No need to repeat for accuracy during the blank parts.
          for i in range(self.repeat_yourself if yes_color else 1):
            point_count += 1
            point_array.extend(self._2B_position(point.x))
            # I'm used to y = 0 meaning the top so that's how I programmed it,
            # but .CTNs put y = 0 at the bottom.
            point_array.extend(self._2B_position(1.0 - point.y))
            if yes_color:
              point_array.extend('\x00\x00\x00\x08')
            else:
              point_array.extend('\x00\x00\x40\x00')
=======
          # Reminder: a point is a namedtuple (x, y) where x and y are between
          # 0.0 and 1.0 inclusive.
          if point_to_point(last_point, point) > MAX_STEP_SIZE:
            raise Exception("This design is invalid because it has 2 points "
                            "that are too far apart.")
          last_point = point
          point_count += 1
          point_array.extend(self._2B_position(point.x))
          # I'm used to y = 0 meaning the top so that's how I programmed it,
          # but .CTNs put y = 0 at the bottom.
          point_array.extend(self._2B_position(1.0 - point.y))
          # TODO: I don't understand how color works exactly. I didn't need
          # to and I got lazy.
          if feature.color > 0:
            point_array.extend('\x00\x00\x00\x08')
          else:
            point_array.extend('\x00\x00\x40\x00')
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34
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
    when value > .5. """
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


class CTN:
  def __init__(self):
    # Again, I'm only ever going to put 1 frame in here. But in the interests
    # of documenting the CTN format, here you go.
    self.frames = []

  def add_frame(self, ctn_frame):
    self.frames.append(ctn_frame)


class PngFeatureFinder:
  def __init__(self, input_file):
    self.input_file = input_file

<<<<<<< HEAD
  def load_features(self, input_file):
    self._load_png(input_file)
    self._find_features()

  def load_from_png(self, input_file):
    self.load_features(input_file)
    self._sort_and_connect_features()
=======
  def get_features(self):
    self._load_png(input_file)
    return self._find_features()
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34

  def _load_png(self):
    self.image = Image.open(self.input_file)
    self.image_width, self.image_height = self.image.size
    if self.image_width > MAX_IMAGE_DIMENSION or \
       self.image_height > MAX_IMAGE_DIMENSION:
      ratio = min([1.0 * MAX_IMAGE_DIMENSION / d for d in self.image.size])
      self.image = self.image.resize((int(ratio * self.image_width), int(ratio * self.image_height)))
      self.image_width, self.image_height = self.image.size
    self.max_image_dimention = max(self.image.size)

  def _find_features(self):
    features = []
    seen_border = Set()
    last_position = None
    for y in range(self.image_height):
      for x in range(self.image_width):
        at = Coord(x, y)
        if not at in seen_border and self._is_border(at):
          seen_pixels, draw_coords = self._compute_feature(at)
          for pixel in seen_pixels:
            seen_border.add(pixel)
          features.append(Feature(255, draw_coords))
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
    return seen_pixels, draw_coords


class DebugFrameDrawer:
  @classmethod
  def draw_debug(cls, frame, output_file_name):
    debug_size = 750
    debug = Image.new("1", (debug_size, debug_size))
    draw = ImageDraw.Draw(debug)
    for feature in cls.features:
      points = [cls._draw_2_debug(c, debug_size) for c in feature.points]
      for index, from_point in enumerate(points):
        to_point = points[(index + 1) % len(points)]
        fx = from_point.x
        fy = from_point.y
        if feature.color > 0:
          draw.line((fx, fy, to_point.x, to_point.y), fill=1)
        draw.line(cls._valid_line(fx - 1, fy, fx + 1, fy, debug_size), fill=1)
        draw.line(cls._valid_line(fx, fy - 1, fx, fy + 1, debug_size), fill=1)
    debug.save(open(output_file_name, "w"), "PNG")

  @classmethod
  def _draw_2_debug(cls, coord, dimension):
    return Coord(int(coord.x * dimension), int(coord.y * dimension))

  @classmethod
  def _valid_line(cls, x1, y1, x2, y2, maximum_exclusive):
    return cls._closest_valid(x1, y1, maximum_exclusive) + \
           cls._closest_valid(x2, y2, maximum_exclusive)

  @classmethod
  def _closest_valid(cls, x, y, maximum_exclusive):
    maximum = maximum_exclusive - 1
    return (min(maximum, max(0, x)), min(maximum, max(0, y)))



class CTNCreator:
  """ Here's where we take our beautiful abstract list of features and turn
  them into gritty CTN files that work OK in real life. """

<<<<<<< HEAD
  def _sane_feature_shuffle(self, feature_list):
    return self.shuffle_features_multiple_lists(
        feature_list, ouput_list_count=1)[0]
=======
  def get_CTNs(self, features, repeat_yourself, max_points):
    for index, feature in features:
      features[index] = self._spread_out_draw_coords(feature)
    # Don't bother repeating points where the laser is off. Accuracy only
    # matters when the laser is on.
    for index, feature in features:
      features[index] = self._repeat_feature(feature, repeat_yourself)

    big_ctns = []
    for feature in features:
      if len(feature.points) > max_points:
        big_ctns.append(self._ctn_from_features([feature]))
        features.remove(feature)

    ctnCount = 1
    ctns = self._get_CTN_helper(features, ctnCount)
    while self._too_many_points(ctns):
      ctnCount += 1
      ctns = self._get_CTN_helper(features, ctnCount)
    return big_ctns + ctns

  def _get_CTN_helper(self, features, ctnCount):
    pass

  def _too_many_points(self, ctns):
    pass

  def _ctn_from_features(self, features):
    pass

  def _repeat_feature(self, feature, repeat_yourself):
    new_coords = []
    for coord in feature.points:
      for i in range(repeat_yourself):
        new_coords.append(coord)
    return Feature(feature.color, new_coords)

  class _FeatureList:
    def __init__(self, first_feature):
      self.features = []
      self.point_count = 0
      self.last_point = None
      self.addFeature(first_feature)

    def addFeature(self, feature):
      self.features.append(feature)
      self.point_count += len(feature.points)

  def _greedy_feature_list_builder(self, features, list_count):
    if len(features) <= list_count:
      return [[f] for f in features]

    feature_copy = list(features)
    feature_lists = []
    list_of_counts = []
    for i in range(list_count):
      random_feature = feature_copy[random.randint(0, len(feature_copy) - 1)]
      feature_copy.remove(random_feature)
      list_of_feature_lists.append(self._FeatureList(random_feature))

    while len(feature_copy) > 0:
      # In terms of python code readability, list comprehensions are the worst
      # thing that ever happened to me.
      shortest = sorted([f.point_count, f for f in feature_lists])[0][1]

  def _sane_feature_shuffle(self, features):
    feature_copy = list(features)
    on = feature_copy[random.randint(0, len(feature_copy) - 1)]
    shuffled = []
    last_point = None
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34

  def shuffle_features_multiple_lists(self, feature_list, ouput_list_count):
    feature_copy = list(feature_list)
    shuffleds = [[] for i in range(ouput_list_count)]

    for shuffled in shuffleds:
      on = feature_copy[random.randint(0, len(feature_copy) - 1)]
      shuffled.append(on)
      feature_copy.remove(on)
    last_points = [None for shuffled in shuffleds]

    while len(feature_copy) > 0:
      # TODO: This is wrong. It should add a feature to whichever list has the
      # least points, not whichever list has the least features.
      for shuffled_index, shuffled in enumerate(shuffleds):
        min_next_feature = None
        min_next_d = sys.maxint
        min_next_feature_point = None
        for feature2 in feature_copy:
          if last_points[shuffled_index]:
            next_feature_point, next_d = self._point_to_feature(
                last_points[shuffled_index], feature2)
          else:
            throw_away, next_feature_point, next_d = \
                self._feature_to_feature_dist(on, feature2)
          if next_d < min_next_d:
            min_next_feature = feature2
            min_next_d = next_d
            min_next_feature_point = next_feature_point
        last_points[shuffled_index] = min_next_feature_point
        on = min_next_feature
        shuffled.append(on)
        feature_copy.remove(on)
    return shuffled

  def _sort_and_connect_features(self, shuffle=True):
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
    account at all.

    This is where we spend most of our time. """
    best_d = sys.maxint
    best_connectors = None
    best_features = self.features
    feature_count = len(self.features)
    for start_fresh in range(5):
      if shuffle:
        feature_copy = self._sane_feature_shuffle(self.features)
      else:
        feature_copy = list(self.features)
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
    # See? Basic high school math. IT'S IMPORTANT!
    return ((to.x - frm.x) ** 2 + (to.y - frm.y) ** 2) ** .5

  def _spread_out_draw_coords(self, feature):
    i = 0
    draw_coords = feature.points
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
    return Feature(feature.color, return_draw_coords)


class CTNFrame:
  def __init__(self, features):
    self.features = features



def main():
  parser = OptionParser()
  # TODO: Keep the concept of CTN files and Frames because those make sense.
  # But don't support multiple input files. Nobody would ever use that without
  # full support, and that's not worth building right now.
  parser.add_option(
      "-i", "--input", dest="input_file",
      help=("A png file. I'll draw a laser path along each border between "
            "white and nonwhite. So for example, a solid black disk will "
            "become a circle around the circumference of the disk."))
  parser.add_option(
      "-o", "--output_file_prefix", dest="output_file_prefix",
      help="Each output CTN file will get this prefix.")
  parser.add_option(
      "-d", "--debug", action="store_true", dest="debug", default=False,
      help="Output a debug .png file for every generated CTN file.")
  parser.add_option(
      "-r", "--repeat_yourself", dest="repeat_yourself", default=3,
      help=("How long should the laser linger as it draws? Higher values "
            "mean more accuracy but more flickering from a slower laser. "
            "Technically, this value says how many times to repeat each point "
            "when the laser is on."))
  parser.add_option(
<<<<<<< HEAD
      "-r", "--repeat_yourself", dest="repeat_yourself", default=2,
      help=("How long should the laser linger at each point. Higher values "
            "mean more accuracy but more flickering."))
  # TODO: This should default to something that makes sense.
  parser.add_option(
      "-m", "--max_points", dest="max_points", default=0,
      help=("How many points per CTN file should we allow? This setting "
            "allows you to break complex designs up into multiple files so "
            "the flickering on each file is bearable."))
=======
      "-m", "--max_points", dest="max_points", default=0,
      help=("A CTN file is a list of points. The laser projector attempts to "
            "point the laser at each point in succession. Use this option to "
            "limit how many points may be in a CTN file before we start "
            "splitting the result across multiple files. This is essential for"
            "complex designs with many points."))
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34
  (options, args) = parser.parse_args()
  output_prefix = options.output_file_prefix
  if output_prefix.lower().endswith(".ctn"):
    output_prefix = output_prefix[:-4]

<<<<<<< HEAD
  input_files = options.input_files.split(",")
  max_points = options.max_points
  if len(input_files) == 0:
    raise Exception("I received no input files.")
  elif len(input_files) > 1 and max_points:
    raise Exception("Either specify more than one input file, or specify "
                    "max_points. It doesn't make sense to specify both.")

  if max_points:
    ctnList = CTNList(max_points=max_points, input_file=input_files[0])
  else:
    ctn = CTN(repeat_yourself=int(options.repeat_yourself))

    for input_file in :
      if not input_file.lower().endswith(".png"):
        raise Exception("This script only accepts .png files as input.")
      ctn_frame = CTNFrame()
      ctn_frame.load_from_png(input_file)
      ctn.add_frame(ctn_frame)
      if options.debug:
        ctn_frame.write_debug(input_file[:-4] + "_debug.png")

    ctn.write(output)
=======
  ctn = CTN(repeat_yourself=int(options.repeat_yourself))

  for input_file in options.input_files.split(","):
    if not input_file.lower().endswith(".png"):
      raise Exception("This script only accepts .png files as input.")
    ctn_frame = CTNFrame()
    ctn_frame.load_from_png(input_file)
    ctn.add_frame(ctn_frame)
    if options.debug:
      ctn_frame.write_debug(input_file[:-4] + "_debug.png")

  ctn.write(output_prefix)
>>>>>>> f7cf128689e69d5e597eb3965bf89950d6462e34


if __name__ == "__main__":
  main()
