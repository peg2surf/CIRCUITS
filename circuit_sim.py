import numpy as np
import math
import itertools

def line_intersection(line1, line2):
    def endpoint(line1: tuple, line2: tuple) -> bool:
        if (line1[0] == line2[0] and line1[1] == line2[1]):
            return True
        if (line1[2] == line2[2] and line1[3] == line2[3]):
            return True
        if (line1[0] == line2[2] and line1[1] == line2[3]):
            return True
        if (line1[2] == line2[0] and line1[3] == line2[1]):
            return True
        return False

    def det(a: tuple, b: tuple):
        return a[0] * b[1] - a[1] * b[0]

    def on_line_sec(line, px, py):
        x1,y1,x2,y2 = line
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

    if endpoint(line1, line2):
        return None
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det((line1[0], line1[1]), (line1[2], line1[3])), det((line2[0], line2[1]), (line2[2], line2[3])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    if on_line_sec(line1, x, y) and on_line_sec(line2, x, y):
        return x, y
    return None


def dot(line1: tuple, line2: tuple):
    l1x = line1[2] - line1[0]
    l1y = line1[3] - line1[1]
    l2x = line2[2] - line2[0]
    l2y = line2[3] - line2[1]
    return (l1x * l2x + l1y * l2y)

def simulate(wire_indexes: np.array, wires: list, sinks: list, out: list = None):
    if out is None:
        out = sinks
    def find_path_iterative(start_idx):
        stack = [(start_idx, [])]  # Stack contains tuples of (current index, path so far)
        paths = []
        visited = set()
        coord_to_idx = {(x, y): i for i, (x, y, _, _) in enumerate(wires)}
        while stack:
            idx, path = stack.pop()
            _, _, b, c = wires[idx]
            new_path = path + [idx]

            if (b, c) in sinks or (b, c) in out:
                paths.append(new_path)
            else:
                for x, y, z, a in wires:
                    if (x, y) == (b, c) and (x, y) not in visited:
                        visited.add((x, y))
                        stack.append((coord_to_idx[(x, y)], new_path))
        return paths

    def line_to_point(line, x, y):
        return math.sqrt((line[0] - x)**2 + (line[1] - y)**2)

    powered = np.zeros(len(wires), int)
    out_power = np.zeros(len(out), int)

    if len(wire_indexes) == 0:
        return powered, out_power

    paths = [path for wire_idx in wire_indexes for path in find_path_iterative(wire_idx)]

    if len(paths) == 0:
        return powered, out_power

    intersects_points = np.zeros((len(wires), len(wires)), dtype=bool)
    L = 0

    while L != len(paths):
        intersected = []
        for i, j in itertools.combinations(set(itertools.chain(*paths)), 2):
            if i == j: continue
            t = line_intersection(wires[i], wires[j])
            if t is None or dot(wires[i], wires[j]) != 0: continue
            x, y = t
            di = line_to_point(wires[i], x, y)
            dj = line_to_point(wires[j], x, y)
            if di <= dj:
                if L == 0:
                    intersects_points[j, i] = True
                intersected.append(j)
            else:
                if L == 0:
                    intersects_points[i, j] = True
                intersected.append(i)
        
        L = len(paths)
         
        resolved = {element for path in paths if not any(element in intersected for element in path) for element in path}
        resolved = {i for i, x in enumerate(intersects_points[:, list(resolved)]) if x.any()}
        paths = [p for p in paths if not any(idx in resolved for idx in p)]

    for x in itertools.chain(*paths):
      powered[x] += 1

    for i, (x, y) in enumerate(out):
      out_power[i] = sum(powered[i] for i, (_, _, a, b) in enumerate(wires) if a==x and b==y)

    return powered, out_power