import numpy as np


def line_search_tool(sampled_traj, x_test, y_test, min_ind):
    #very mindless search tool
    # inputs:
    # x_test, y_test is the point that you want to compare to the trajectory
    # sampled_traj is a sampled version of the trajectory
    # min_ind is an index of the sampled trajector from which you want to do a forward search
    assert len(sampled_traj.shape) == 2
    assert sampled_traj.shape[0] == 2

    best_dist = float('inf')
    circular_index = min_ind
    for i in range(0, sampled_traj.shape[1]):
        if circular_index + i >= sampled_traj.shape[1]:
            circular_index = -i
        dist = (x_test - sampled_traj[0, int(circular_index + i)])**2 + (
            y_test - sampled_traj[1, int(circular_index + i)])**2
        if dist < best_dist:
            best_dist = dist
        else:
            break

    if circular_index + i - 1 == -1:
        closest_ind = int(sampled_traj.shape[1] - 1)
    else:
        closest_ind = int(circular_index + i - 1)
    closest_pt = sampled_traj[:, closest_ind]
    return closest_pt[0], closest_pt[1], closest_ind


def wrap_angle(rad):
    # converts angles outside of +/-PI to +/-PI
    if (rad > np.pi):
        rad -= 2 * np.pi
    if (rad < -np.pi):
        rad += 2 * np.pi
    return rad
