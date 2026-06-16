def recursive_xy_cut(boxes, index_list, direction='X') -> list:
    if len(index_list) <= 1:
        return index_list
        
    if direction == 'X':
        index_list = sorted(index_list, key=lambda idx: boxes[idx]['coords'][0])
    else:
        index_list = sorted(index_list, key=lambda idx: boxes[idx]['coords'][1])
        
    split_idx = -1
    max_gap = -1
    
    for i in range(len(index_list) - 1):
        idx1 = index_list[i]
        idx2 = index_list[i+1]
        box1 = boxes[idx1]['coords']
        box2 = boxes[idx2]['coords']
        
        if direction == 'X':
            gap = box2[0] - box1[2]
        else:
            gap = box2[1] - box1[3]
            
        if gap > max_gap:
            max_gap = gap
            split_idx = i
            
    if max_gap > 10:
        left_part = index_list[:split_idx+1]
        right_part = index_list[split_idx+1:]
        next_direction = 'Y' if direction == 'X' else 'X'
        return recursive_xy_cut(boxes, left_part, next_direction) + recursive_xy_cut(boxes, right_part, next_direction)
    else:
        if direction == 'X':
            return recursive_xy_cut(boxes, index_list, 'Y')
        else:
            return sorted(index_list, key=lambda idx: (boxes[idx]['coords'][1], boxes[idx]['coords'][0]))

def sort_boxes_by_xy_cut(boxes: list) -> list:
    initial_indices = list(range(len(boxes)))
    return recursive_xy_cut(boxes, initial_indices, 'X')
