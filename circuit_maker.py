from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import sys
import numpy as np
import itertools
import math
from statistics import mean

from circuit_sim import simulate
from magic_circles import circle_hash, CIRCLE_HASH_TO_NAME

pygame.init()

WIDTH, HEIGHT = 800, 600
CELL_SIZE = 50
screen = pygame.display.set_mode((WIDTH + 100, HEIGHT + 100))
pygame.display.set_caption("Wire Maker")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
LIGHTBLUE = 0, 255, 255
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
PINK = (255,20,147)

x_offset = 40
y_offset = 20

sources = []
sinks = []
sinks_colors = []
shapes = []
shapes_colors = []
shapes_connections = []
wires = []
wires_colors = []

CIRCLEMAP = {
    3: (57, 255, 20),  #Neon Green
    4: (255, 255, 0),  #Neon Yellow
    5: (0, 191, 255),  #Neon Blue
    6: (255, 0, 71),   #Neon Red
    7: (150, 155, 160),#Neon Gray
    8: (255, 255, 230),#Neon White
}
PRESETCOMPS = [
    None,
    None,
    None,
    {(4, 5), (3, 4), (3, 5)},
    {(6, 7), (4, 5), (5, 6), (4, 7)},
    {(5, 9), (6, 7), (8, 9), (5, 6), (7, 8)},
    {(9, 10), (10, 11), (6, 11), (6, 7), (8, 9), (7, 8)},
    {(9, 10), (10, 11), (7, 13), (12, 13), (11, 12), (8, 9), (7, 8)},
    {(9, 10), (13, 14), (10, 11), (8, 15), (12, 13), (11, 12), (8, 9), (14, 15)},
]

def find_loops(coordinates):
    graph = {}

    # Build the graph
    for x1, y1, x2, y2 in coordinates:
        if (x1, y1) not in graph:
            graph[(x1, y1)] = []
        if (x2, y2) not in graph:
            graph[(x2, y2)] = []
        graph[(x1, y1)].append((x2, y2))

    def dfs(node, visited, path):
        if node in visited:
            return path
        visited.add(node)
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, visited, path)
        return path

    set_loops = []
    loops = []
    visited = set()
    for node in graph:
        if node not in visited:
            loop = dfs(node, set(), [node])
            set_loop = set(loop)
            if len(loop) > 2 and loop[0] == loop[-1] and set_loop not in set_loops:
                set_loops.append(set_loop)
                loops.append(loop)
    return loops

def run():
    global sources
    global shape
    global sinks
    global wires
    global sinks_colors
    global wires_colors
    global shapes_colors

    inv = []
    subset = []
    for i, shape in enumerate(shapes):
        subset.append(PRESETCOMPS[len(shape) - 1].issubset(shapes_connections[i]))
        if subset[i]:
            inv.append(shape)
    
    o = np.fromiter((
        j for s in itertools.chain(sources, *inv) for j, t in enumerate(wires) 
        if s[0] == t[0] and s[1] == t[1] and (len(s) != 3 or s[2])
    ), dtype=int)
    S = sinks + [x for i, shape in enumerate(shapes) if not subset[i] for x in shape[:-1]]
    val, pow = simulate(o, wires, S)
    for i, x in enumerate(val):
        if x:
            wires_colors[i] = LIGHTBLUE
        else:
            wires_colors[i] = RED
    for i, x in enumerate(pow[:len(sinks)]):
        if x != 0:
            sinks_colors[i] = GREEN
        else:
            sinks_colors[i] = RED
    counter = len(sinks)
    for i, shape in enumerate(shapes):
        c = len(shape) - 1
        if subset[i] or any(pow[counter: counter+c]):
            shapes_colors[i] = CIRCLEMAP[c]
        else:
            shapes_colors[i] = GRAY
        counter += c
             
t = None
shape_t = None

def pos_adj(pos: tuple):
    return (round(pos[0]/CELL_SIZE), round(pos[1]/CELL_SIZE))

def is_point_on_line(x, y, x1, y1, x2, y2):
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

#, size=CELL_SIZE/50, color = color
def render_arrowhead(x, y, z, a, size: int = 1, color: tuple = GRAY):
    if (x+line_thick < 0 and z+line_thick < 0) or (x-line_thick > WIDTH and z-line_thick > WIDTH): return
    if (y < 0 and a < 0) or (y > HEIGHT and a > HEIGHT): return
    # Calculate midpoint
    mid_x = (x + z) // 2
    mid_y = (y + a) // 2
    
    # Calculate angle between start and end points
    dx = z - x
    dy = a - y
    angle = math.atan2(dy, dx)
    
    # Set arrowhead size
    arrow_size = 10*size
    
    # Calculate arrowhead points
    arrow_points = [
        (mid_x - arrow_size * math.cos(angle - math.pi / 6), mid_y - arrow_size * math.sin(angle - math.pi / 6)),
        (mid_x, mid_y),  # Midpoint
        (mid_x - arrow_size * math.cos(angle + math.pi / 6), mid_y - arrow_size * math.sin(angle + math.pi / 6))
    ]
    
    # Draw arrowhead
    pygame.draw.polygon(screen, color, arrow_points)


def wire_draw():
    global t
    global points
    global wires
    if t is None:
        pos = event.pos
        pos = (pos[0] - x_offset, pos[1] - y_offset)
        t = pos_adj((pos[0], pos[1]))
    else:
        pos = event.pos
        pos = (pos[0] - x_offset, pos[1] - y_offset)
        wire = (*t, *pos_adj((pos[0], pos[1])))
        if wire[0] == wire[2] and wire[1] == wire[3]: return
        if wire in wires: return
        wires.append(wire)
        wires_colors.append(BLUE)
        t = None

def wire_passive():
    global t
    if t is not None:
        x, y = t
        pos = pygame.mouse.get_pos()
        pos_x = round((pos[0]-x_offset)/CELL_SIZE)*CELL_SIZE + x_offset
        pos_y = round((pos[1]-y_offset)/CELL_SIZE)*CELL_SIZE + y_offset
        x = x*CELL_SIZE + x_offset
        y = y*CELL_SIZE + y_offset
        pygame.draw.line(screen, BLUE, (x, y), (pos_x, pos_y), 2)
        render_arrowhead(x, y, pos_x, pos_y, color=BLUE)

def wire_delete():
    global wires
    pos = event.pos
    pos = (pos[0] - x_offset, pos[1] - y_offset)
    pos = pos_adj(pos)
    x, y = pos
    for i, wire in enumerate(wires):
        if is_point_on_line(x, y, *wire):
            wires.pop(i)
            wires_colors.pop(i)

def sources_draw():
    global sources
    pos = event.pos
    pos = (pos[0] - x_offset, pos[1] - y_offset)
    pos = pos_adj(pos)
    if pos in sources: return
    sources.append(pos)

def sources_passive(): 
    pos = pygame.mouse.get_pos()
    pos = (
        round((pos[0]-x_offset)/CELL_SIZE)*CELL_SIZE + x_offset, 
        round((pos[1]-y_offset)/CELL_SIZE)*CELL_SIZE + y_offset,
    )
    pygame.draw.circle(screen, LIGHTBLUE, pos, max(10*(CELL_SIZE/50), 1))

def sources_delete():
    global sources
    pos = event.pos
    pos = (pos[0] - x_offset, pos[1] - y_offset)
    pos = pos_adj(pos)
    if pos in sources:
        sources.remove(pos)
        

def sinks_draw():
    global sinks
    global sinks_colors
    pos = event.pos
    pos = (pos[0] - x_offset, pos[1] - y_offset)
    pos = pos_adj(pos)
    if pos in sinks: return
    sinks.append(pos)
    sinks_colors.append(GRAY)

def sinks_passive():
    pos = pygame.mouse.get_pos()
    pos = (
        round((pos[0]-x_offset)/CELL_SIZE)*CELL_SIZE + x_offset, 
        round((pos[1]-y_offset)/CELL_SIZE)*CELL_SIZE + y_offset,
    )
    pygame.draw.circle(screen, RED, pos, max(10*(CELL_SIZE/50), 1))

def sinks_delete():
    global sinks
    global sinks_colors
    pos = event.pos
    pos = (pos[0] - x_offset, pos[1] - y_offset)
    pos = pos_adj(pos)
    if pos in sinks:
        i = sinks.index(pos)
        sinks.pop(i)
        sinks_colors.pop(i)

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_closest_point(point, points, *ignore):
    min_distance = float('inf')
    closest_point = None
    for i, p in enumerate(points):
        if i in ignore: continue
        distance = euclidean_distance(point, p)
        if distance < min_distance:
            min_distance = distance
            closest_point = (i, p)
    return closest_point

def avg(iterator):
        return mean(x for x,_ in iterator)*CELL_SIZE + x_offset, mean(y for _,y in iterator)*CELL_SIZE + y_offset
def midpoints(loop):
    for (x1, y1), (x2,y2) in zip(loop, loop[1:]):
        yield ( (x1+x2)//2, (y1+y2)//2 )

def are_all_values_used(values, L):
    return PRESETCOMPS[L].issubset(set(values))

def shape_wire(shape, points_):
    global shape_t
    pos = event.pos
    L = len(shapes[shape]) -1
    if shape_t is None:
        all_vals = are_all_values_used(set(shapes_connections[shape]), L)
        if not all_vals:
            i, _ = find_closest_point(pos, points_[:L*2])
        else:
            i, _ = find_closest_point(pos, points_[L:])
            i = i + L
        shape_t = i
    else:
        i = shape_t
        if 2*L > i  >= L:
            j, _ = find_closest_point(pos, points_[L:L*2])
            j += L
        elif i >= 2*L:
            j, _ = find_closest_point(pos, points_[L*2:])
            j += L*2
        else:
            j, _ = find_closest_point(pos, points_[:L])
            if abs(i-j) == 1:
                return
        val = (min(i, j), max(i, j))
        if val not in shapes_connections[shape]:
            shapes_connections[shape].add(val)
            shape_t = None
        
passive_render = []
def shape_wire_passive(shape, points_):
    global shape_t
    if shape_t is not None:
        pos = pygame.mouse.get_pos()
        i = shape_t
        p = points_[i]
        if 2*L > i  >= L:
            _, end = find_closest_point(pos, points_[L:L*2])
        elif i >= 2*L:
            _, end = find_closest_point(pos, points_[L*2:])
        else:
            _, end = find_closest_point(pos, points_[:L])
        passive_render.append((GRAY, p, end))
        pygame.draw.line(screen, GRAY, p, end, 2)

def shape_wire_delete(shape, points_):
    def distance_point_to_line(a, b, x1, y1, x2, y2):
        a_line = y2 - y1
        b_line = x1 - x2
        c_line = x2 * y1 - x1 * y2        
        numerator = abs(a_line * a + b_line * b + c_line)
        denominator = math.sqrt(a_line**2 + b_line**2)
        distance = numerator / denominator
        return distance
    global shape_t
    pos = event.pos
    if len(shapes_connections[shape]) == 0:
        shapes.pop(shape)
        shapes_colors.pop(shape)
        shapes_connections.pop(shape)
        rects.remove(shape)
    else:
        _, p = min(((distance_point_to_line(*pos, *points_[i], *points_[j]), (i,j)) for i, j in shapes_connections[shape]), key=lambda x: x[0])
        shapes_connections[shape].remove(p)
        

def no_action(*_):
    pass

sel = 0

options = {
    "Wire"  : [wire_draw, wire_passive, wire_delete, shape_wire, shape_wire_passive, shape_wire_delete],
    "Source": [sources_draw, sources_passive, sources_delete, no_action, no_action, no_action], 
    "Sink"  : [sinks_draw, sinks_passive, sinks_delete, no_action, no_action, no_action],
    "Circle": [no_action, no_action, no_action, no_action, no_action, no_action],
}

option_name = tuple(options.keys())
l_options = len(options)

right_button_offset_x = 0
right_button_offset_y = 0
right_button_pos = None
right_button_pressed = False

def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_convex(points):
    n = len(points)
    if n < 3:
        return False

    sign = None
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = cross_product(p1, p2, p3)
        if sign is None or cross * sign >= 0:
            sign = cross
        else:
            return False
    return True

def circle_hash_api(L, conections):
    def has_repeating_number(list_of_tuples):
        seen_numbers = set()
        for tuple in list_of_tuples:
            for number in tuple:
                if number in seen_numbers:
                    return True
                seen_numbers.add(number)
        return False
    def both_values_within_range(val1: float, val2: float, lower: float, upper: float) -> bool:
        return (lower <= val1 < upper and lower <= val2 < upper) or not(lower <= val1 < upper or lower <= val2 < upper)
    def intersections(parent, slashes, L):
        out = []
        for s,d in slashes:
            for e,f in parent:
                e = e%L
                f = f%L
                lower = min(e, f)
                upper = max(e, f)
                if not both_values_within_range(s%L, d%L, lower, upper):
                    out.append((s, d))
                    break
        return out

    main = [(a, b) for a, b in conections if a<L]
    layer1 = [(a, b) for a, b in conections if 2*L > a >= L]
    layer2 = [(a, b) for a, b in conections if  a >= 2*L]
    if len(layer1) >= L and not has_repeating_number(layer2):
        o = intersections([(a, b) for a, b in layer1 if (a,b) not in PRESETCOMPS[L]], layer2, L)
        return f"-{circle_hash(L, o)}"
    elif not has_repeating_number(layer1):
        o = intersections(main, layer1, L)
        return circle_hash(L, o)

def render_text(text: str, color: tuple, position: tuple, center:bool = False, font_size: int = None):
    if text is None:
        return
    if font_size is None:
        font = pygame.font.Font(None, int((40/50)*CELL_SIZE))
    else:
        font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = position
    else:
        text_rect.topleft = position
    (x1, y1), (x2, y2) = text_rect.topleft, text_rect.bottomright
    if x2 < 0 or x1 > WIDTH or y2 < 0 or y1 > HEIGHT: return
    screen.blit(text_surface, text_rect)

def render_line(screen, color, x, y, z, a, line_thick):
    if (x+line_thick < 0 and z+line_thick < 0) or (x-line_thick > WIDTH and z-line_thick > WIDTH): return
    if (y < 0 and a < 0) or (y > HEIGHT and a > HEIGHT): return
    pygame.draw.line(screen, color, (x, y), (z, a), line_thick)

def render_circle(screen, color, x, y, dot_r):
    if x+dot_r < 0 or x-dot_r > WIDTH or y+dot_r < 0 or y-dot_r > HEIGHT: return
    pygame.draw.circle(screen, color, (x, y), dot_r)

def render_polygon(screen, color, shape_index, points):
    if all(x < 0 or x > WIDTH or y < 0 or y > HEIGHT for x, y in points): return
    rects.add(shape_index)
    pygame.draw.polygon(screen, color, points)


help = False
delete_mode = False

rects = set()

def ray_casting(x, y, polygon):
    num_vertices = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

while True:
    screen.fill(BLACK)
    pygame.draw.line(screen, GRAY, (0, HEIGHT), (WIDTH, HEIGHT))
    pygame.draw.line(screen, GRAY, (WIDTH, 0), (WIDTH, HEIGHT))
    for x in range(x_offset % CELL_SIZE, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(y_offset % CELL_SIZE, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

    intersect_shape = -1
    points_ = None
    for i in rects:
        _t = [(x*CELL_SIZE + x_offset, y*CELL_SIZE + y_offset) for x, y in shapes[i]]
        if ray_casting(*pygame.mouse.get_pos(), _t):
            intersect_shape = i
            m_ = list(midpoints(_t))
            points_ = _t[:-1] + m_ + list(midpoints(m_ + m_[0:1]))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and delete_mode:
                if intersect_shape == -1:
                    options[option_name[sel]][2]()
                else:
                    options[option_name[sel]][5](intersect_shape, points_)
            elif event.button == 1:  # Left mouse button
                if intersect_shape == -1:
                    options[option_name[sel]][0]()
                else:
                    options[option_name[sel]][3](intersect_shape, points_)
            if event.button == 3:
                if not right_button_pressed:
                    right_button_offset_x = x_offset
                    right_button_offset_y = y_offset
                    right_button_pos = event.pos
                    right_button_pressed = True
            if event.button == 4:  # Scroll up
                CELL_SIZE += 10
            elif event.button == 5:  # Scroll down
                CELL_SIZE = max(CELL_SIZE - 10, 5)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Right mouse button
                right_button_pressed = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                delete_mode = True
            if event.key == pygame.K_ESCAPE:
                t = None
                shape_t = None
            if event.key == pygame.K_r:
                run()
            if event.key == pygame.K_t:
                help = not help
            if event.unicode.isdigit():
                o = (int(event.unicode) - 1)%10
                if o < l_options and o != sel:
                    t = None
                    sel = o
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_d:
                delete_mode = False
        if event.type == pygame.MOUSEMOTION and right_button_pressed:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            x_offset = mouse_x - right_button_pos[0] + right_button_offset_x
            y_offset = mouse_y - right_button_pos[1] + right_button_offset_y


    for loop in find_loops(wires):
        if len(loop) > 9 or not is_convex(loop): continue
        for (x1, y1), (x2, y2) in zip(loop, loop[1:]):
            i = wires.index((x1, y1, x2, y2))
            wires.pop(i)
            wires_colors.pop(i)
        shapes.append(loop)
        shapes_colors.append(BLUE)
        shapes_connections.append(set())


    line_thick = max(int(2*(CELL_SIZE/50)), 1)
    dot_r = max(10*(CELL_SIZE/50), 1)

    opt = option_name[sel]
    if (not delete_mode and intersect_shape == -1) or (opt == "Wire" and t is not None):
        options[opt][1]()
    elif not delete_mode:
        options[opt][4](intersect_shape, points_)

    def _t(iter):
        (x1, y1), (x2, y2) = iter
        return ((x1+x2)/2, (y1+y2)/2)
    
    for (x, y, z, a), color in zip(wires, wires_colors):
        x = x*CELL_SIZE + x_offset
        z = z*CELL_SIZE + x_offset
        y = y*CELL_SIZE + y_offset
        a = a*CELL_SIZE + y_offset
        render_line(screen, color, x, y, z, a, line_thick) 
        render_arrowhead(x, y, z, a, size=CELL_SIZE/50, color = color)
    for x, y in sources:
        x = x*CELL_SIZE + x_offset
        y = y*CELL_SIZE + y_offset
        render_circle(screen, LIGHTBLUE, x, y, dot_r)
    for (x, y), color in zip(sinks, sinks_colors):
        x = x*CELL_SIZE + x_offset
        y = y*CELL_SIZE + y_offset
        render_circle(screen, color, x, y, dot_r)

    for i, (shape, color, connections) in enumerate(zip(shapes, shapes_colors, shapes_connections)):
        L = len(shape) - 1
        render_lines = []
        render_circles = []
        points = []
        midpoint = []
        for (x, y), (z, a) in zip(shape, shape[1:]):
            x = x*CELL_SIZE + x_offset
            z = z*CELL_SIZE + x_offset
            y = y*CELL_SIZE + y_offset
            a = a*CELL_SIZE + y_offset
            points.append((x, y))
            midpoint.append(((x+z)//2, (y+a)//2))
            render_circles.append((x, y))
            render_lines.append(((x, y), (z, a)))
        midpoint2 = list(midpoints(midpoint + midpoint[0:1]))
        for k, j in shapes_connections[i]:
            if 2*L > k >= L:
                cord1 = midpoint[k%L]
                cord2 = midpoint[j%L]
            elif k >= 2*L:
                cord1 = midpoint2[k%L]
                cord2 = midpoint2[j%L]
            else:
                cord1 = points[k]
                cord2 = points[j]
            render_lines.append((cord1, cord2))
        #if intersect_shape != i:
        render_polygon(screen, BLACK, i, points)
        for (x, y), (z, a) in render_lines:
            render_line(screen, color, x, y, z, a, line_thick)
        for (x, y) in render_circles:
            render_circle(screen, color, x, y, dot_r)
        for color, (x, y), (z, a) in passive_render:
            render_line(screen, color, x, y, z, a, line_thick)
            passive_render.clear()
        
        render_text(CIRCLE_HASH_TO_NAME.get(circle_hash_api(L, shapes_connections[i])), GRAY, avg(shape[:-1]), True)
    
    render_text(f"Current Tool: {option_name[sel]}{' delete' if delete_mode else ''}", GRAY, (0, 0), font_size=40)

    if help:
        render_text(f"Actions", GRAY, (0, 50), font_size=40)
        for i, element in enumerate(option_name):
            render_text(f"{i+1} = {element}", GRAY, (0, 50 + (i+1) * 40), font_size=40)
        i+=2
        render_text(f"d (hold) = delete mode for tool", GRAY, (0, 50 + i * 40), font_size=40)
        i+=1
        render_text(f"r = run", GRAY, (0, 50 + i * 40), font_size=40)
        i+=1
        render_text(f"t = toggle help", GRAY, (0, 50 + i * 40), font_size=40)
        i+=1
        render_text(f"wheel = zoom in/out", GRAY, (0, 50 + i * 40), font_size=40)
        i+=1
        render_text(f"left click = move screen", GRAY, (0, 50 + i * 40), font_size=40)
        

    pygame.display.flip()