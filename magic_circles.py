import numpy as np
import math
from functools import cache
from collections import Counter
import itertools

# NODE MATH
def both_values_within_range(val1: float, val2: float, lower: float, upper: float) -> bool:
  return (lower <= val1 <= upper and lower <= val2 <= upper) or not(lower <= val1 <= upper or lower <= val2 <= upper)

def compare_coords(a: tuple, b: tuple):
  return a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]

def intersects(diag1: tuple, diag2: tuple) -> bool:  
  return compare_coords(diag1, diag2) or not both_values_within_range(diag1[0], diag1[1], min(diag2[0], diag2[1]), max(diag2[0], diag2[1]))

def midpoints_intersects(midpoint: tuple, main: tuple)-> bool:
  return not both_values_within_range(midpoint[0] + 0.5, midpoint[1] + 0.5, min(main[0], main[1]), max(main[0], main[1]))

def triple_intersects(midpoint1: tuple, midpoint2: tuple, main: tuple)-> bool:
  return intersects(midpoint1, midpoint2) and midpoints_intersects(midpoint1, main) and midpoints_intersects(midpoint1, main)

def double_intersect(a: tuple, b: tuple, d: tuple, e: tuple) -> bool:
  return midpoints_intersects(a, d) and midpoints_intersects(b, d) and midpoints_intersects(a, e) and midpoints_intersects(b, e)

def dist(a, b, mod):
  c = abs(a - b)
  return min(c, mod - c) % mod
# NODE MATH END

def valid_circle(cords: list):
  return len(set(itertools.chain(*cords))) == len(cords) * 2 
   

def circle_hash(parent: int, cords: list):
  return "".join(map(str,(parent,
      *(a for x in sorted(
          Counter(
              [dist(a, b, parent) for a, b in cords]
          ).items(), key=lambda x: x[0]
      ) for a in x)
  )))

CIRCLE_HASH_TO_NAME = {
   "3"      : "Enhance",
   "-3"      : "Anti-Enhance",
   "4"      : "Lock",
   "-4"      : "Unlock",
   "411"    : "Expand",
   "-411"   : "Contract",
   "412"    : "Cut",
   "-412"   : "Repair",
   "421"    : "Twist",
   "-421"    : "Counter Twist",
   "5"      : "Stasis",
   "-5"     : "Anti-Stasis",
   "511"    : "Translation Out",
   "-511"   : "Translation In",
   "512"    : "Translation Lock",
   "-512"   : "Translation Unlock",
   "521"    : "Rotate",
   "-521"   : "Counter Rotate",
   "51121"  : None,
   "6"      : "Heat",
   "-6"     : "Heat Absorb",
   "611"    : "Infra-Red WL",
   "-611"   : "Infra-Red WL Absorb",
   "612"    : "Visible-Light",
   "-612"   : "Visible-Light Absorb",
   "613"    : "Ultra-Violet WL",
   "-613"   : "Ultra-Violet WL Absorb",
   "621"    : "Sound",
   "-621"   : "Sound Absorb",
   "622"    : "Tremor",
   "-622"   : "Tremor Absorb",
   "631"    : "Electricity",
   "-631"   : "Electricity Absorb",
   "61121"  : "Explosion",
   "-61121" : "Explosion Absorb",
   "61231"  : None,
   "7"      : "Unstruct",
   "-7"     : "Restruct",
   "8"      : "Construct",
   "-8"     : "Destruct"
}

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  @cache
  def create_circle(radius: float, center=(0, 0), num_points: int=100):
      # Create a circle using NumPy
      angles = np.linspace(0, 2 * np.pi, num_points)
      x = center[0] + radius * np.cos(angles)
      y = center[1] + radius * np.sin(angles)
      return x, y

  def make_cords_equal(cords, tolerance=0.1):
      """
      Modify the coordinates to make either the x or y coordinates equal if they are close enough.

      Args:
      cords (list of tuples): List of coordinates in the format [(x1, y1), (x2, y2), ...].
      tolerance (float): Tolerance level for closeness. Defaults to 0.1.

      Returns:
      list of tuples: Modified list of coordinates.
      """
      modified_cords = []

      for x, y in cords:
          close_x = any(abs(x - other_x) < tolerance for other_x, _ in modified_cords)
          close_y = any(abs(y - other_y) < tolerance for _, other_y in modified_cords)

          if close_x:
              modified_cords.append((modified_cords[-1][0], y))
          elif close_y:
              modified_cords.append((x, modified_cords[-1][1]))
          else:
              modified_cords.append((x, y))

      return modified_cords

  @cache
  def get_polygon_coordinates(n: int, radius:float, offset:float=math.pi/2, center:tuple=(0, 0)) -> list:
      # Calculate the angle between each vertex
      angle = 2 * math.pi / n
      # Calculate coordinates of the vertices
      coordinates = []
      for i in range(n):
          x = center[0] + radius * math.cos(i * angle + offset)
          y = center[1] + radius * math.sin(i * angle + offset)
          coordinates.append((x, y))
      return coordinates

  def get_midpoints(coordinates: list) -> list:
      midpoints = []
      num_vertices = len(coordinates)
      for i in range(num_vertices):
          current_vertex = coordinates[i]
          next_vertex = coordinates[(i + 1) % num_vertices]
          mid_x = (current_vertex[0] + next_vertex[0]) / 2
          mid_y = (current_vertex[1] + next_vertex[1]) / 2
          midpoints.append((mid_x, mid_y))
      return midpoints

  R = 7
  N = 6

  cir_x, cir_y = create_circle(R)
  poly = get_polygon_coordinates(N, R)
  
  x = [a for a, s in poly]
  x_ = x + [x[0]]
  y = [s for a, s in poly]
  y_ = y + [y[0]]
  midpoint = get_midpoints(poly)
  x_midpoint = [a for a, s in midpoint]
  x_midpoint_ = x_midpoint + [x_midpoint[0]]
  y_midpoint = [s for a, s in midpoint]
  y_midpoint_ = y_midpoint + [y_midpoint[0]]

  cords = [
      (0,1),
      (2,5),
      (3,4),
  ]

  print(
      valid_circle(cords)
  )

  print(
      circle_hash(N, cords)
  )

  fig, axs = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
  axs.plot(cir_x, cir_y)
  axs.plot(x_, y_)
  for a, b in cords:
      axs.plot((x_midpoint[a], x_midpoint[b]), (y_midpoint[a], y_midpoint[b]))
  axs.axis('equal')
  axs.plot()
  plt.show()
  while True:
      pass
  #for i in range(N):
  #    fig, axs = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
  #    axs.plot(*cir)
  #    axs.plot(x_, y_)
  #    o = [ ((x + i)%N, (y + i)%N) for x, y in cords ]
  #    #print( circle_hash(N, o) )
  #    for a, b in o:
  #       axs.plot((x_midpoint[a], x_midpoint[b]), (y_midpoint[a], y_midpoint[b]))
  #    axs.axis('equal')
  #    axs.plot()
  #plt.show()
  #for i in range(N):
  #    fig, axs = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
  #    axs.plot(*cir)
  #    axs.plot(x_, y_)
  #    for a, b in [(i, (5+i)%N)]:
  #        axs.plot((x_midpoint[a], x_midpoint[b]), (y_midpoint[a], y_midpoint[b]))
  #    axs.axis('equal')
  #    axs.plot()
  #plt.axis('equal')
  #plt.show()
