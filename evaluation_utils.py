import numpy as np
import cv2

########## Use for frame processing
def crop_and_resize (frame):
    return cv2.resize(frame[36:250, 114:328, :], (15, 15), interpolation = cv2.INTER_LINEAR)

def get_source_from_2_frames( prev_frame, cur_frame ):
    diff = abs(prev_frame.astype(np.int32) - cur_frame.astype(np.int32))
    vals = np.argwhere(diff >= 20)
    
    points = set()
    for val in vals:
        points.add(tuple(val[:2]))
    
    # if prev_frame[points[0]] ~ (255,255,255), points[0] is target, points[1] is start
    # otherwise reverse
    
    points = list(points)
    
    if np.sum(abs(prev_frame[points[0]].astype(np.int32) - np.array([255, 255, 255]))) > 20:
        source = np.array(points[0])
    else:

        source = np.array(points[1])
        
    return source

def move ( loc, action ):
    if action == 'left':
        return (loc[0], loc[1] - 1)
    if action == 'right':
        return (loc[0], loc[1] + 1)
    if action == 'up':
        return (loc[0] - 1, loc[1])
    if action == 'down':
        return (loc[0] + 1, loc[1])
    
def reverse_action ( action ):
    if action == 'left':
        return 'right'
    if action == 'right':
        return 'left'
    if action == 'up':
        return 'down'
    if action == 'down':
        return 'up'

def reenact( video_file, actions ):
    """
    Reading the first frame of video_file, 
    perform actions on the purple blocks, and regenerate the correct list of actions
    """
    cap = cv2.VideoCapture(video_file)
    
    # First 2 frames
    _, first_frame = cap.read()
    first_frame = crop_and_resize ( first_frame )
    _, second_frame = cap.read()
    second_frame = crop_and_resize ( second_frame )
    
    source = get_source_from_2_frames ( first_frame, second_frame )
    
    # Clear first_frame[tuple(source)]
    first_frame[tuple(source)] = [255, 255, 255]
    
    loc = source
    
    new_actions = []
    for action in actions:
        new_loc = move (loc, action)
        
        # Check if new_loc is legal move
        if 0 <= new_loc[0] <  first_frame.shape[0] and 0 <= new_loc[1] <  first_frame.shape[1]:
            # Check if new_loc is not occupied
            if np.sum(abs(first_frame[tuple(new_loc)].astype(np.int32) - np.array([255, 255, 255]))) > 20:
                # Not empty
                pass
            else:
                loc = new_loc
                new_actions.append(action)
    
    return new_actions

def neighbor_score ( chain1, chain2):
    """
    Calculating neighboring scores from chain1 to chain2
    This code could be implemented with 4 heaps, but only efficient if the length of chain1, chain2 are large enough
    One heap for only positive pairs
    One heap for only negative pairs
    One heap for (positive, negative)
    One heap for (negative, positive)
    One remaining list for pair that has one value == 0 (that could change to one of the aforementioned heaps)
    
    However here we just do a very simple loop in this code
    """
    values = [(0,0)]
    for action in chain1:
        values.append ( move (values[-1], action) )
    
    shortest_vals = []
    for action in chain2:
        r_action = reverse_action ( action )
        
        values = [ move(value, r_action) for value in values]
        shortest_val = np.min( [abs(value[0]) + abs(value[1]) for value in values] )
        shortest_vals.append(shortest_val)
        
    return shortest_vals
    
    
def shortest_neighbor_score ( refs,  preds, video_file ):
    correct_preds = reenact( video_file, preds )
    
    score_1 = neighbor_score( refs , correct_preds )
    score_2 = neighbor_score( correct_preds, refs)
    
    total_val = sum(score_1) + sum(score_2)
    total_count = ( 1 # First cell
                + np.count_nonzero(score_1)
                + len(score_2) ) # score_1 and score_2 share cells that has value == 0

    # print ("===================")
    # print (score_1)
    # print (score_2)
    # print ('%.3f' % (total_val / total_count))
    return total_val / total_count, len(score_1), len(score_2)