import random
import numpy as np
import math
#from generate_shapes import check_in_frame, generate_rect_shape, \
#    generate_triag_shape, generate_l_shape 

def check_in_frame ( size, pos ):
    """
    Allow space around the shape
    """
    if 0 <= pos[0] < size and  0 <= pos[1] < size:
        return True
    return False

def check_in_frame_with_space ( size, pos ):
    """
    Allow space around the shape
    """
    if 1 <= pos[0] < size - 1 and  1 <= pos[1] < size - 1:
        return True
    return False

def generate_rect_shape(frame, position, height, width , color):
    """
    Parameters:
    =====================
    frame: squared frame
    width: width of rectangle
    height: height of rectangle
    position: tuple of 2
    color: (color the cells with this value)
    
    Returns:
    =====================
    - success: whether imposition successes or not
    """
    size = frame.shape[0]
    other_corner = ( position[0] + height - 1, position[1] + width - 1 )
    
    if not check_in_frame ( size, position ) or not check_in_frame ( size, other_corner ):
        return False
    
    frame[ position[0] : position[0] + height, 
          position[1] : position[1] + width] = np.ones((height, width)) * color
    
    return True

def generate_triag_shape(frame, position, side, direction, color):
    """
    Impose an triangle shape structure on the frame, position is where you start it (top-left corner of rectangle)
    
    This is a triangle with side = 4, direction = 0
    
    S
    o o 
    o o o
    o o o o
    
    direction = 1
    
    S o o o
    o o o 
    o o
    o 
    
    direction = 2
    
    S o o o
      o o o 
        o o
          o 
    
    direction = 3
    
    S     o
        o o 
      o o o
    o o o o
    
    Parameters:
    =====================
    frame: squared frame
    side: length of the shorter side of triangle
    direction: 
    position: tuple of 2
    color: (color the cells with this value)
    
    Returns:
    =====================
    - success: whether imposition successes or not
    
    """
    size = frame.shape[0]
    other_corner = ( position[0] + side - 1, position[1] + side - 1 )
    
    if not check_in_frame ( size, position) or not check_in_frame ( size, other_corner ):
        return False
    
    inner_shape = np.tril(np.ones((side, side)) * color)
    
    # Generate for direction == 0, and rotate it to the correct direction
    rotated_shape = np.rot90 ( inner_shape, -direction )
    
    
    frame[ position[0] : position[0] + rotated_shape.shape[0], 
          position[1] : position[1] + rotated_shape.shape[1] ] = rotated_shape
    
    return True

def generate_l_shape(frame, position, height, width, direction, color):
    """
    Impose an triangle shape structure on the frame, position is where you start it (top-left corner of rectangle)
    
    This is a triangle with width = 4, height = 3, direction = 0
    
    S
    o 
    o o o o
    
    direction = 1
    
    S o o o
    o 
    o
    
    direction = 2
    
    S o o o
          o 
          o
    
    direction = 3
    
    S     o
          o 
    o o o o
    
    Parameters:
    =====================
    frame: squared frame
    width: width of rectangle
    height: height of rectangle
    direction: 
    position: tuple of 2
    color: (color the cells with this value)
    
    Returns:
    =====================
    - success: whether imposition successes or not
    
    """
    size = frame.shape[0]
    other_corner = ( position[0] + height - 1, position[1] + width - 1 )
    
    if not check_in_frame ( size, position) or not check_in_frame ( size, other_corner ):
        return False
    
    if direction == 0 or direction == 3:
        frame[ position[0] + height - 1,  position[1] : position[1] + width ] = color
    
    if direction == 1 or direction == 2:
        frame[ position[0]             ,  position[1] : position[1] + width ] = color
        
    if direction == 0 or direction == 1:
        frame[ position[0]: position[0] + height,  position[1] ] = color
        
    if direction == 2 or direction == 3:
        frame[ position[0]: position[0] + height,  position[1] + width - 1 ] = color
    
    return True

def point_in_rect ( point , rect ):
    """
    point: tuple
    rect: (pos_x, pos_y, height, width)
    """
    pos_x, pos_y, height, width = rect
    if pos_x <= point[0] <= pos_x + height and pos_y <= point[1] <= pos_y + width:
        return True
    return False

def overlapping ( rect1, rect2 ) :
    """
    Check if two rects overlap or not

    rect of form (pos_x, pos_y, height, width)
    """

    if rect2[0] < rect1[0]:
        # swap
        rec = rect1
        rect1 = rect2
        rect2 = rec

    # at this point rect1[0] <= rect2[0]
    return point_in_rect( (rect1[0], rect1[1] + rect1[3] - 1), rect2 ) or \
        point_in_rect( (rect1[0] + rect1[2] - 1, rect1[1] + rect1[3] - 1), rect2 ) or \
        point_in_rect( (rect2[0], rect2[1]), rect1 ) or \
        point_in_rect( (rect2[0] + rect2[2] - 1, rect2[1]), rect1 )


def generate_rectangles ( size, no_of_rect, no_of_square, rect_range, sqrs_range ):
    """
    On a frame of size x size

    Generate non-overlapping no_of_rect rectangles and no_of_square squares

    Returns a list of rectangles: (pos_x, pos_y, height, width)
    and list of squares: (pos_x, pos_y, size)

    This algorithm is not an exact algorithm, it might generate less than the
    intended number of rects and squares.

    Parameters:
    ===============
    size: 
    no_of_rect: 
    no_of_square:
    rect_range: (a, b) -> a <= rect size <= b
    sqrs_range: (a, b) -> a <= rect size <= b

    Returns:
    ===============
    results: 
      - results[0] = [ (pos_x, pos_y, height, width) ]
      - results[1] = [ (pos_x, pos_y, size ) ]

    """
    rects = []
    for _ in range(10 * no_of_rect):
        height = random.randint( rect_range[0], rect_range[1] )
        width = random.randint( rect_range[0], rect_range[1] )

        pos_x = random.randint( 0, size - height )
        pos_y = random.randint( 0, size - width )

        new_rect = (pos_x, pos_y, height, width)

        for r in rects:
            if distance (new_rect, r) == 0:
                break
            # if overlapping(r, new_rect) :
            #     break
        else:
            rects.append(new_rect)

        if len(rects) == no_of_rect:
            break

    sqrs_rects = []
    sqrs = []
    for _ in range(10 * no_of_square):
        sqr_size = random.randint( sqrs_range[0], sqrs_range[1] )
        pos_x = random.randint( 0, size - sqr_size )
        pos_y = random.randint( 0, size - sqr_size )

        new_rect = (pos_x, pos_y, sqr_size, sqr_size)
        new_sqr = (pos_x, pos_y, sqr_size)

        for r in rects:
            if distance (new_rect, r) == 0:
                break
            # if overlapping(r, new_rect) :
            #     break
        else:
            for r in sqrs_rects:
                if distance (new_rect, r) == 0:
                    break
                # if overlapping(r, new_rect) :
                #     break
            else:
                sqrs_rects.append(new_rect)
                sqrs.append(new_sqr)

        if len(sqrs) ==  no_of_square:
            break

    return (rects, sqrs)

def distance ( rect1 , rect2 ):
    """
    rect1 and rect2 have the following forms (pos_x, pos_y, height, width)
    """
    pos_x_1, pos_y_1, height_1, width_1 = rect1
    pos_x_2, pos_y_2, height_2, width_2 = rect2
    
    center_1 = (pos_x_1 + height_1/2, pos_y_1 + width_1/2)
    center_2 = (pos_x_2 + height_2/2, pos_y_2 + width_2/2)
    
    diff = (abs(center_1[0] - center_2[0]), abs(center_1[1] - center_2[1]))
    
    # min ( max (0,x_diff), max (0,y_diff))
    distance =  max (0, diff[0] - height_1/2 - height_2/2 ) + max (0, diff[1] - width_1/2 - width_2/2 )
    
    return distance

def distance_to_edge ( rect, width, height ):
    pos_x_1, pos_y_1, height_1, width_1 = rect
    
    center_1 = (pos_x_1 + height_1/2, pos_y_1 + width_1/2)
    
    diff = max (0, np.min ([ pos_x_1, pos_y_1, height - pos_x_1 - height_1, width - pos_y_1 - width_1 ]))
    
    return diff

def generate_rectangles_sparse ( size, no_of_rect, no_of_square, rect_range, sqrs_range ):
    """
    On a frame of size x size

    Generate non-overlapping no_of_rect rectangles and no_of_square squares

    Returns a list of rectangles: (pos_x, pos_y, height, width)
    and list of squares: (pos_x, pos_y, size)

    This algorithm is not an exact algorithm, it might generate less than the
    intended number of rects and squares.

    Parameters:
    ===============
    size: 
    no_of_rect: 
    no_of_square:
    rect_range: (a, b) -> a <= rect size <= b
    sqrs_range: (a, b) -> a <= rect size <= b

    Returns:
    ===============
    results: 
      - results[0] = [ (pos_x, pos_y, height, width) ]
      - results[1] = [ (pos_x, pos_y, size ) ]

    """
    rects = []
    
    # Let's call t[shape_index] is the sum of the mean distance to the closest shape + distance to the edge 
    #        of frame
    # This is a qualitative value measuring how sparse a shape is in regards to other shapes
    t = np.zeros(no_of_rect + no_of_square)
    
    for _ in range(5 * no_of_rect):
        height = random.randint( rect_range[0], rect_range[1] )
        width = random.randint( rect_range[0], rect_range[1] )

        pos_x = random.randint( 0, size - height )
        pos_y = random.randint( 0, size - width )

        new_rect = (pos_x, pos_y, height, width)
        
        if len(rects) < no_of_rect:
            for r in rects:
                if overlapping(r, new_rect) :
                    break
            else:
                rect_index = len(rects)
                # Calculate t[rect_index]

                de = distance_to_edge(new_rect, size, size) 
                de = 0

                if rect_index == 0:
                    t[rect_index] = de
                else:
                    t[rect_index] = np.min([distance (new_rect, r) for r in rects]) + de
                
                # Recalculate t[r]
                for rect_index, r in enumerate(rects):
                    de = distance_to_edge(r, size, size)
                    de = 0
                    t[rect_index] = min(t[rect_index], distance (new_rect, r) + de)
                
                rects.append(new_rect)
        elif len(rects) == no_of_rect:
            print (t[:no_of_rect])
            """
            We try to replace one of the previous rect if this rect makes a sparser maze
            
            We want one that has distance to the closest rect larger, but we don't want the 
            shapes to be all on the edges.
            """
            ds = [distance (new_rect, r) for r in rects]
            de = distance_to_edge(new_rect, size, size)
            de = 0
            for r_index in range(no_of_rect):
                u = np.min([d for i, d in enumerate(ds) if i != r_index]) + de
                if u > t[r_index]:
                    t[r_index] = u
                    rects[r_index] = new_rect
                    break

    sqrs_rects = []
    sqrs = []
    for _ in range(4 * no_of_square):
        sqr_size = random.randint( sqrs_range[0], sqrs_range[1] )
        pos_x = random.randint( 0, size - sqr_size )
        pos_y = random.randint( 0, size - sqr_size )

        new_rect = (pos_x, pos_y, sqr_size, sqr_size)
        new_sqr = (pos_x, pos_y, sqr_size)

        for r in rects:
            if overlapping(r, new_rect) :
                break
        else:
            for r in sqrs_rects:
                if overlapping(r, new_rect) :
                    break
            else:
                sqrs_rects.append(new_rect)
                sqrs.append(new_sqr)

        if len(sqrs) ==  no_of_square:
            break

    return (rects, sqrs)

def visualize ( size , rects, sqrs ):
    frame = np.zeros((size, size))

    for i, rect in enumerate(rects):
        pos_x, pos_y, height, width = rect
        frame[pos_x : pos_x + height, pos_y : pos_y + width] = i + 1

    for j, sqr in enumerate(sqrs):
        pos_x, pos_y, size = sqr
        frame[pos_x : pos_x + size, pos_y : pos_y + size] = i + j + 2

    return frame


def generate_frame ( size, no_of_rect, no_of_l, no_of_trig, rect_range ):
    """
    We can just use shape functions, and imposing each shape on the frame, one by one

    - generate_l_shape ( frame, pos, height, width, direction, color )

    - generate_rect_shape ( frame, pos, height, width, color )

    - generate_triag_shape ( frame, pos, size, direction, color )

    """
    rects, sqrs = generate_rectangles ( size, no_of_rect + no_of_l, \
        no_of_trig, rect_range, rect_range)

    frame = np.zeros((size, size))

    # Impose
    for i, rect in enumerate(rects[:no_of_rect]):
        pos_x, pos_y, height, width = rect
        frame[pos_x : pos_x + height, pos_y : pos_y + width] = i + 1

    for j, rect in enumerate(rects[no_of_rect:]):
        pos_x, pos_y, height, width = rect
        direction = random.randint(0, 3)
        generate_l_shape (frame, (pos_x, pos_y), height, width, direction, i + j + 2 )

    for k, sqr in enumerate(sqrs):
        pos_x, pos_y, size = sqr
        direction = random.randint(0, 3)
        generate_triag_shape (frame, (pos_x, pos_y), size, direction, i + j + k + 3)

    return frame


def generate_src_target ( frame, threshold = None ):
    """
    Generate source and target satisfy distance threshold

    We want the source and target to be faily distanced so that the path
    between them might be rather complex
    """
    size = frame.shape[0]
    if threshold is None:
        threshold = size

    while True:
        a, b, c, d = np.random.randint(0, size - 1, 4)
        if frame[a,b] == 0 and frame[c,d] == 0 and abs(a - c) + abs(b - d) >= threshold:
            return (a,b), (c,d)

def get_neighbors ( size, pos ):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 or j == 0) and not i == j == 0:
                new_pos = (pos[0] + i, pos[1] + j)
                if check_in_frame ( size, new_pos ):
                    neighbors.append(new_pos)
    return neighbors

def get_neighbors_random ( size, pos ):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 or j == 0) and not i == j == 0:
                new_pos = (pos[0] + i, pos[1] + j)
                if check_in_frame ( size, new_pos ):
                    neighbors.append(new_pos)

    return [tuple(t) for t in np.random.permutation(neighbors)]

def generate_path ( frame, source, target, exception_vals = [], neighbor_func = get_neighbors_random ) :
    """
    Generate path movement between a source and a target
    This algorithm depends on neighbor_func
    If neighbor_func is deterministic, it would be deterministic,
    because there are multiple
    shortest paths between two points. 

    Parameters
    ===========
    frame: value map
    source: 
    target:
    exceptions_vals: (optional) by default, we can only traverse through 0 cell,
        adding values into this list allow you to cut through shapes
    get_neighbors: lambda (size, pos) -> list of neighbor pos

    Returns
    ===========
    path: list of points, not including the source and target, 
        if there is no path return None
    """
    size = frame.shape[0]
    # Keep track of the previous cell that has been expanded to
    parent = {}
    
    ### There is nothing too difficult here, we run overflow algorithm to cover the space
    f = frame.copy()
    f[source] = -2
    f[target] = -3
    
    explore_list = [ source ]
    
    def search ( l ):
        while l:
            # Get the shortest path to the target
            new_explore_list = []

            for pos in l:
                for n in neighbor_func( size, pos ):
                    if f[n] == -3:
                        # Found
                        parent[n] = pos
                        return
                    if f[n] == 0 or f[n] in exception_vals:
                        # Add into new list
                        new_explore_list.append(n)
                        parent[n] = pos
                        # 4 is marked
                        f[n] = -4
            l = new_explore_list
        
    
    search ( explore_list )
    cur = target
    
    path = []
    while cur in parent:
        path.append(cur)
        cur = parent[cur]

    if len(path) == 0:
        # No path
        return None

    # Remove first and last state
    return path[-1:0:-1]

def shortest_path ( frame, shape1_index, shape2_index ):
    """
    Shortest path between two shapes

    This is an approximate algorithm that generate 
    a shortest path from a point of shape1 to a point of shape2

    Parameters
    ============
    frame: 
    shape1_index: the values of the pixels corresponding to shape1
    shape2_index: the values of the pixels corresponding to shape2

    Returns
    ============
    - list of points between p1 and p2 if path exists, otherwise None.
        if p1 and p2 is next to each other, return []
    """
    shape1_indices = np.where( frame == shape1_index )
    shape2_indices = np.where( frame == shape2_index )
    if len(shape1_indices[0]) == 0 or len(shape2_indices[0]) == 0:
        return None

    # Top left corners of two shapes
    p1 = (shape1_indices[0][0], shape1_indices[1][0])
    p2 = (shape2_indices[0][0], shape2_indices[1][0])

    path = generate_path (frame, p1, p2, exception_vals = [shape1_index, shape2_index],
        neighbor_func = get_neighbors)

    # new_path is path that only keeps point outside of two shapes
    new_path = []

    for p in path:
        if frame[p] == shape1_index:
            new_path = []
        elif frame[p] == shape2_index:
            return new_path
        else:
            new_path.append(p)
    return new_path
