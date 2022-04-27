import cv2
import ctypes
import numpy as np

import inspect
from os.path import abspath, dirname, join

fname = abspath(inspect.getfile(inspect.currentframe()))
lib = ctypes.cdll.LoadLibrary(join(dirname(fname), 'astar.so'))

astar = lib.astar
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
astar.restype = ctypes.c_bool
astar.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                  ndmat_i_type]


weighted_astar = lib.weighted_astar
weighted_astar.restype = ctypes.c_bool
weighted_astar.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                           ctypes.c_float, ctypes.c_int, ndmat_i_type]


multi_goal_astar = lib.multi_goal_astar
multi_goal_astar.restype = ctypes.c_int
multi_goal_astar.argtypes = [ndmat_f_type, ndmat_f_type, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ndmat_i_type,
                             ctypes.c_int, ctypes.c_bool, ndmat_i_type]


multi_goal_weighted_astar = lib.multi_goal_weighted_astar
multi_goal_weighted_astar.restype = ctypes.c_int
multi_goal_weighted_astar.argtypes = [ndmat_f_type, ndmat_f_type, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ndmat_i_type,
                             ctypes.c_int, ctypes.c_bool, ctypes.c_float,
                             ctypes.c_int, ndmat_i_type]


def astar_path(obmap, start, goal, allow_diagonal=False):
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= obmap.shape[0] or
            start[1] < 0 or start[1] >= obmap.shape[1]):
        raise ValueError('Start of (%d, %d) lies outside grid.' % (start))
    # Ensure goal is within bounds.
    if (goal[0] < 0 or goal[0] >= obmap.shape[0] or
            goal[1] < 0 or goal[1] >= obmap.shape[1]):
        raise ValueError('Goal of (%d, %d) lies outside grid.' % (goal))

    height, width = obmap.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    success = astar(
        obmap.flatten(), height, width, start_idx, goal_idx, allow_diagonal,
        paths  # output parameter
    )
    if not success:
        return np.array([])

    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])

def astar_planner(obmap, start, goal, allow_diagonal=False):
    """
    start - (x, y) coordinates
    goal - (x, y) coordinates

    Returns:
        path_x, path_y - a list of x, y coordinates
                         starting from GOAL to the START
    """
    # astar_path requires (y, x) as input
    path_start = (start[1], start[0])
    path_goal = (goal[1], goal[0])
    path = astar_path(obmap, path_start, path_goal, allow_diagonal=allow_diagonal)

    if path.shape[0] > 0:
        path_y = path[:, 0].tolist()[::-1]
        path_x = path[:, 1].tolist()[::-1]
    else:
        path_y = None
        path_x = None

    return path_x, path_y

def weighted_astar_path(obmap, start, goal, allow_diagonal=False,
                        wscale=4.0, niters=1):
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= obmap.shape[0] or
            start[1] < 0 or start[1] >= obmap.shape[1]):
        raise ValueError('Start of (%d, %d) lies outside grid.' % (start))
    # Ensure goal is within bounds.
    if (goal[0] < 0 or goal[0] >= obmap.shape[0] or
            goal[1] < 0 or goal[1] >= obmap.shape[1]):
        raise ValueError('Goal of (%d, %d) lies outside grid.' % (goal))
    # Ensure niters >= 1
    assert(niters >= 1)

    height, width = obmap.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    success = weighted_astar(
        obmap.flatten(), height, width, start_idx, goal_idx,
        allow_diagonal, wscale, niters,
        paths  # output parameter
    )
    if not success:
        return np.array([])

    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])

def weighted_astar_planner(obmap, start, goal, allow_diagonal=False,
                           wscale=4.0, niters=1):
    """
    start - (x, y) coordinates
    goal - (x, y) coordinates

    Returns:
        path_x, path_y - a list of x, y coordinates
                         starting from GOAL to the START
    """
    # weighted_astar_path requires (y, x) as input
    path_start = (start[1], start[0])
    path_goal = (goal[1], goal[0])
    path = weighted_astar_path(obmap, path_start, path_goal,
                               allow_diagonal=allow_diagonal,
                               wscale=wscale, niters=niters)

    if path.shape[0] > 0:
        path_y = path[:, 0].tolist()[::-1]
        path_x = path[:, 1].tolist()[::-1]
    else:
        path_y = None
        path_x = None

    return path_x, path_y

def multi_goal_astar_path(obmap, gmap, start, goals, allow_diagonal=False):
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= obmap.shape[0] or
            start[1] < 0 or start[1] >= obmap.shape[1]):
        raise ValueError('Start of (%d, %d) lies outside grid.' % (start))

    height, width = obmap.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idxs = []
    for g_y, g_x in zip(*goals):
        goal_idx = np.ravel_multi_index((int(g_y), int(g_x)), (height, width))
        goal_idxs.append(goal_idx)
    goal_idxs = np.array(goal_idxs, dtype=np.int32)

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    reached_goal_idx = multi_goal_astar(
        obmap.flatten(), gmap.flatten(), height, width, start_idx,
        goal_idxs, len(goal_idxs), allow_diagonal,
        paths  # output parameter
    )
    if reached_goal_idx == -1:
        return np.array([])

    coordinates = []
    path_idx = reached_goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])

def multi_goal_astar_planner(
    obmap, start, gmap, allow_diagonal=False, use_contours=False
):
    """
    start - (x, y) coordinates
    goal - (x, y) coordinates

    Returns:
        path_x, path_y - a list of x, y coordinates
                         starting from GOAL to the START
    """
    # astar_path requires (y, x) as input
    path_start = (start[1], start[0])
    if not use_contours:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (gmap * 255).astype(np.uint8), 8
        )
        assert n_labels > 1, "MultiGoalAstar: goal map is empty!"
        path_goals = (centroids[1:, 1], centroids[1:, 0])
    else:
        contours = cv2.findContours(
            (gmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        path_goals_y, path_goals_x = [], []
        for contour in contours:
            contour = contour[0][:, 0]
            if len(contour.shape) == 1:
                continue
            path_goals_y.append(contour[:, 1])
            path_goals_x.append(contour[:, 0])
        path_goals = (np.concatenate(path_goals_y, axis=0),
                      np.concatenate(path_goals_x, axis=0))
    path = multi_goal_astar_path(obmap, gmap, path_start, path_goals, allow_diagonal=allow_diagonal)

    if path.shape[0] > 0:
        path_y = path[:, 0].tolist()[::-1]
        path_x = path[:, 1].tolist()[::-1]
    else:
        path_y = None
        path_x = None

    return path_x, path_y


def multi_goal_weighted_astar_path(obmap, gmap, start, goals,
                                   allow_diagonal=False, wscale=4.0, niters=1):
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= obmap.shape[0] or
            start[1] < 0 or start[1] >= obmap.shape[1]):
        raise ValueError('Start of (%d, %d) lies outside grid.' % (start))

    height, width = obmap.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idxs = []
    for g_y, g_x in zip(*goals):
        goal_idx = np.ravel_multi_index((int(g_y), int(g_x)), (height, width))
        goal_idxs.append(goal_idx)
    goal_idxs = np.array(goal_idxs, dtype=np.int32)

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    reached_goal_idx = multi_goal_weighted_astar(
        obmap.flatten(), gmap.flatten(), height, width, start_idx,
        goal_idxs, len(goal_idxs), allow_diagonal, wscale, niters,
        paths  # output parameter
    )
    if reached_goal_idx == -1:
        return np.array([])

    coordinates = []
    path_idx = reached_goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])


def multi_goal_weighted_astar_planner(obmap, start, gmap, allow_diagonal=False,
                                      use_contours=False, wscale=4.0, niters=1):
    """
    start - (x, y) coordinates
    goal - (x, y) coordinates

    Returns:
        path_x, path_y - a list of x, y coordinates
                         starting from GOAL to the START
    """
    # astar_path requires (y, x) as input
    path_start = (start[1], start[0])
    if not use_contours:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (gmap * 255).astype(np.uint8), 8
        )
        assert n_labels > 1, "MultiGoalAstar: goal map is empty!"
        path_goals = (centroids[1:, 1], centroids[1:, 0])
    else:
        contours = cv2.findContours(
            (gmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        path_goals_y, path_goals_x = [], []
        for contour in contours:
            contour = contour[0][:, 0]
            if len(contour.shape) == 1:
                continue
            path_goals_y.append(contour[:, 1])
            path_goals_x.append(contour[:, 0])
        path_goals = (np.concatenate(path_goals_y, axis=0),
                      np.concatenate(path_goals_x, axis=0))

    path = multi_goal_weighted_astar_path(
        obmap, gmap, path_start, path_goals, allow_diagonal=allow_diagonal,
        wscale=wscale, niters=niters,
    )

    if path.shape[0] > 0:
        path_y = path[:, 0].tolist()[::-1]
        path_x = path[:, 1].tolist()[::-1]
    else:
        path_y = None
        path_x = None

    return path_x, path_y
