MANO_LEFT_KEYPOINTS = [
    'left_wrist', 'left_index_1', 'left_index_2', 'left_index_3',
    'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_pinky_1',
    'left_pinky_2', 'left_pinky_3', 'left_ring_1', 'left_ring_2',
    'left_ring_3', 'left_thumb_1', 'left_thumb_2', 'left_thumb_3',
    'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky'
]

MANO_RIGHT_KEYPOINTS = [
    'right_wrist', 'right_index_1', 'right_index_2', 'right_index_3',
    'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_pinky_1',
    'right_pinky_2', 'right_pinky_3', 'right_ring_1', 'right_ring_2',
    'right_ring_3', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
]

MANO_LEFT_REORDER_KEYPOINTS = [
    'left_wrist', 'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb',
    'left_index_1', 'left_index_2', 'left_index_3', 'left_index',
    'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_middle',
    'left_ring_1', 'left_ring_2', 'left_ring_3', 'left_ring', 'left_pinky_1',
    'left_pinky_2', 'left_pinky_3', 'left_pinky'
]

MANO_RIGHT_REORDER_KEYPOINTS = [
    'right_wrist', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3',
    'right_thumb', 'right_index_1', 'right_index_2', 'right_index_3',
    'right_index', 'right_middle_1', 'right_middle_2', 'right_middle_3',
    'right_middle', 'right_ring_1', 'right_ring_2', 'right_ring_3',
    'right_ring', 'right_pinky_1', 'right_pinky_2', 'right_pinky_3',
    'right_pinky'
]

# Deprecated: reserved for backward compatibility
MANO_KEYPOINTS = MANO_RIGHT_KEYPOINTS
# Two hands (left + right)
MANO_HANDS_KEYPOINTS = MANO_LEFT_KEYPOINTS + MANO_RIGHT_KEYPOINTS
# Reordered two hands (left + right)
MANO_HANDS_REORDER_KEYPOINTS = \
        MANO_LEFT_REORDER_KEYPOINTS + MANO_RIGHT_REORDER_KEYPOINTS
