import networkx as nx
import numpy as np
from scipy.signal import savgol_filter
from enum import Enum

from typing import List, Union, Tuple, Generator, Set
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet, LaneletType, LaneletNetwork
from commonroad.scenario.laneletcomplement import make_curve

# modified based on https://commonroad.in.tum.de/route-planner
def resample_polyline(polyline: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resamples the input polyline with the specified step size.

    The distances between each pair of consecutive vertices are examined. If it is larger than the step size,
    a new sample is added in between.

    :param polyline: polyline with 2D points
    :param step: minimum distance between each consecutive pairs of vertices
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return np.array(polyline)

    polyline_new = [polyline[0]]

    current_idx = 0
    current_position = step
    current_distance = np.linalg.norm(polyline[0] - polyline[1])

    # iterate through all pairs of vertices of the polyline
    while current_idx < len(polyline) - 1:
        if current_position <= current_distance:
            # add new sample and increase current position
            ratio = current_position / current_distance
            polyline_new.append((1 - ratio) * polyline[current_idx] +
                                ratio * polyline[current_idx + 1])
            current_position += step

        else:
            # move on to the next pair of vertices
            current_idx += 1
            # if we are out of vertices, then break
            if current_idx >= len(polyline) - 1:
                break
            # deduct the distance of previous vertices from the position
            current_position = current_position - current_distance
            # compute new distances of vertices
            current_distance = np.linalg.norm(polyline[current_idx + 1] - polyline[current_idx])

    # add the last vertex
    polyline_new.append(polyline[-1])

    return np.array(polyline_new)

class RouteType(Enum):
    # Survival routes have no specific goal lanelet
    REGULAR = "regular"
    SURVIVAL = "survival"

class Route:
    """Class to represent a route in the scenario."""

    def __init__(self, lanelet_network: LaneletNetwork, planning_problem: PlanningProblem, list_ids_lanelets: List[int],
                 route_type: RouteType, set_ids_lanelets_permissible: Set[int] = None):

        self.planning_problem = planning_problem
        self.lanelet_network = lanelet_network

        # a route is created given the list of lanelet ids from start to goal
        self.list_ids_lanelets = list_ids_lanelets
        self.type = route_type

        # a section is a list of lanelet ids that are adjacent to a lanelet in the route
        self.list_sections = list()
        self.set_ids_lanelets_in_sections = set()
        self.set_ids_lanelets_opposite_direction = set()

        if set_ids_lanelets_permissible is None:
            self.set_ids_lanelets_permissible = {lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets}
        else:
            self.set_ids_lanelets_permissible = set_ids_lanelets_permissible

        # generate reference path from the list of lanelet ids leading to goal
        self.reference_path_segments, self.reference_lanelets, self.principle_reference = self._generate_reference_path()
        self.reference_paths = [make_curve(seg) for seg in self.reference_path_segments]
        #self.ind = 0
        #self.list_ids_lanelets_route = []
        #for lanelet_ids in self.reference_lanelets:
        #    for lanelet_id in lanelet_ids:
        #        if lanelet_id not in self.list_ids_lanelets_route:
        #            self.list_ids_lanelets_route.append(lanelet_id)

    def _generate_reference_path(self) -> np.ndarray:
        """Generates a reference path (polyline) out of the given route

        This is done in four steps:
        1. compute lane change instructions
        2. compute the portion of each lanelet based on the instructions
        3. compute the reference path based on the portion
        4. smoothen the reference path

        :return: reference path in 2d numpy array ([[x0, y0], [x1, y1], ...])
        """
        instruction = self._compute_lane_change_instructions()
        # add a segments
        return self._compute_reference_segments(instruction)
        #list_portions = self._compute_lanelet_portion(instruction)
        #reference_path = self._compute_reference_path(list_portions)
        #reference_path_smoothed = chaikins_corner_cutting(reference_path)

    def _compute_lane_change_instructions(self) -> List[int]:
        """Computes lane change instruction for planned routes

        The instruction is a list of 0s and 1s, with 0 indicating  no lane change is required
        (driving straight forward0, and 1 indicating that a lane change (to the left or right) is required.
        """
        list_instructions = []
        for idx, id_lanelet in enumerate(self.list_ids_lanelets[:-1]):
            if self.list_ids_lanelets[idx + 1] in self.lanelet_network.find_lanelet_by_id(id_lanelet).successor:
                list_instructions.append(0)
            else:
                list_instructions.append(1)

        # add 0 for the last lanelet
        list_instructions.append(0)

        return list_instructions

    def _compute_reference_segments(self, list_instructions, step_resample=1.0):
        #list_instructions_consecutive = [list(v) for k, v in itertools.groupby(list_instructions)]
        reference_segments = []
        reference_lanelets = []
        reference_from_to = []
        dict_ids_lanelets = {}
        num_lanelets_in_route = len(self.list_ids_lanelets)
        for idx, id_lanelet in enumerate(self.list_ids_lanelets):
            lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet)
            # resample the center vertices to prevent too few vertices with too large distances
            vertices_resampled = resample_polyline(lanelet.center_vertices, step_resample)
            num_vertices = len(vertices_resampled)

            if idx == 0:
                if list_instructions[idx] == 0:
                    # straight
                    di = np.array([[0.0] for _ in range(len(vertices_resampled))])
                else:
                    # merge
                    next_lanelet = self.lanelet_network.find_lanelet_by_id(self.list_ids_lanelets[idx+1])
                    #w2 = np.sum((lanelet.center_vertices[0] - next_lanelet.center_vertices[0])**2)
                    #wi = np.array([[np.sqrt(w2)] for _ in range(len(vertices_resampled))])
                    if lanelet.adj_left == next_lanelet.lanelet_id:
                        di = np.array([[1.0] for _ in range(len(vertices_resampled))])
                    else:
                        di = np.array([[-1.0] for _ in range(len(vertices_resampled))])
                lanelet_leftmost = lanelet
                while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                    lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

                lanelet_rightmost = lanelet
                while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                    lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

                if lanelet.stop_line is not None:
                    #start = lanelet.stop_line.start
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                    ind = np.argmin(np.sqrt(np.sum((vertices_resampled - lanelet.stop_line.start)**2, axis=1)))
                    sl[ind, 0] = 1.0
                else:
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                reference_segments.append(np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1))
                reference_lanelets.append([id_lanelet for _ in range(len(vertices_resampled))])
                reference_from_to.append([np.array([0, len(vertices_resampled)])])
            elif list_instructions[idx-1] == 0:
                if list_instructions[idx] == 0:
                    di = np.array([[0.0] for _ in range(len(vertices_resampled))])
                else:
                    next_lanelet = self.lanelet_network.find_lanelet_by_id(self.list_ids_lanelets[idx+1])
                    #w2 = np.sum((lanelet.center_vertices[0] - next_lanelet.center_vertices[0])**2)
                    #wi = np.array([[np.sqrt(w2)] for _ in range(len(vertices_resampled))])
                    if lanelet.adj_left == next_lanelet.lanelet_id:
                        di = np.array([[1.0] for _ in range(len(vertices_resampled))])
                    else:
                        di = np.array([[-1.0] for _ in range(len(vertices_resampled))])
                lanelet_leftmost = lanelet
                while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                    lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

                lanelet_rightmost = lanelet
                while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                    lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

                if lanelet.stop_line is not None:
                    #start = lanelet.stop_line.start
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                    ind = np.argmin(np.sqrt(np.sum((vertices_resampled - lanelet.stop_line.start)**2, axis=1)))
                    sl[ind, 0] = 1.0
                else:
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                path_to_be_concatenated = np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1)
                reference_segments[-1] =  np.concatenate((reference_segments[-1], path_to_be_concatenated[1:]), axis=0)
                reference_lanelets[-1].extend([id_lanelet for _ in range(len(path_to_be_concatenated)-1)])
                reference_from_to[-1].append(np.array([len(reference_segments[-1])-len(vertices_resampled)+1, len(reference_segments[-1])]))
            else:
                if list_instructions[idx] == 0:
                    di = np.array([[0.0] for _ in range(len(vertices_resampled))])
                else:
                    next_lanelet = self.lanelet_network.find_lanelet_by_id(self.list_ids_lanelets[idx+1])
                    #w2 = np.sum((lanelet.center_vertices[0] - next_lanelet.center_vertices[0])**2)
                    #wi = np.array([[np.sqrt(w2)] for _ in range(len(vertices_resampled))])
                    if lanelet.adj_left == next_lanelet.lanelet_id:
                        di = np.array([[1.0] for _ in range(len(vertices_resampled))])
                    else:
                        di = np.array([[-1.0] for _ in range(len(vertices_resampled))])
                lanelet_leftmost = lanelet
                while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                    lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

                lanelet_rightmost = lanelet
                while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                    lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

                if lanelet.stop_line is not None:
                    #start = lanelet.stop_line.start
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                    ind = np.argmin(np.sqrt(np.sum((vertices_resampled - lanelet.stop_line.start)**2, axis=1)))
                    sl[ind, 0] = 1.0
                else:
                    sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                reference_segments.append(np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1))
                reference_lanelets.append([id_lanelet for _ in range(len(vertices_resampled))])
                reference_from_to.append([np.array([0, len(vertices_resampled)])])

            dict_ids_lanelets[id_lanelet] = len(reference_segments)-1
            if lanelet.adj_left is not None and lanelet.adj_left_same_direction and lanelet.adj_left not in dict_ids_lanelets:
                dict_ids_lanelets[lanelet.adj_left] = len(reference_segments)-1
            if lanelet.adj_right is not None and lanelet.adj_right_same_direction and lanelet.adj_right not in dict_ids_lanelets:
                dict_ids_lanelets[lanelet.adj_right] = len(reference_segments)-1

        lanelet_start = self.lanelet_network.find_lanelet_by_id(self.list_ids_lanelets[0])
        if lanelet_start.adj_left is not None and lanelet_start.adj_left not in self.list_ids_lanelets:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_start.adj_left)
            vertices_resampled = resample_polyline(lanelet.center_vertices, step_resample)
            di = np.array([[-1.0] for _ in range(len(vertices_resampled))])
            lanelet_leftmost = lanelet
            while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
            lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

            lanelet_rightmost = lanelet
            while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
            lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

            if lanelet.stop_line is not None:
                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                ind = np.argmin(np.sqrt(np.sum((vertices_resampled - lanelet.stop_line.start)**2, axis=1)))
                sl[ind, 0] = 1.0
            else:
                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
            segment = np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1)

            reference_segments.insert(0, segment)
            reference_lanelets.insert(0, [lanelet_start.adj_left for _ in range(len(vertices_resampled))])

        if lanelet_start.adj_right is not None and lanelet_start.adj_right not in self.list_ids_lanelets:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_start.adj_right)
            vertices_resampled = resample_polyline(lanelet.center_vertices, step_resample)
            di = np.array([[1.0] for _ in range(len(vertices_resampled))])
            lanelet_leftmost = lanelet
            while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
            lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

            lanelet_rightmost = lanelet
            while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
            lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

            if lanelet.stop_line is not None:
                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                ind = np.argmin(np.sqrt(np.sum((vertices_resampled - lanelet.stop_line.start)**2, axis=1)))
                sl[ind, 0] = 1.0
            else:
                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
            segment = np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1)

            reference_segments.insert(0, segment)
            reference_lanelets.insert(0, [lanelet_start.adj_right for _ in range(len(vertices_resampled))])


        for i in reversed(range(len(reference_lanelets)-1)):
            id_lanelets, segment  = reference_lanelets[i], reference_segments[i]
            id_last_lanelet = id_lanelets[-1]
            last_lanelets = [self.lanelet_network.find_lanelet_by_id(id_last_lanelet)]
            while len(last_lanelets) > 0:
                last_lanelet = last_lanelets.pop(0)
                if len(last_lanelet.successor) > 0:
                    for id_suc in last_lanelet.successor:
                        if id_suc in dict_ids_lanelets and id_suc not in self.list_ids_lanelets:
                            if id_suc in id_lanelets:
                                continue
                            successor = self.lanelet_network.find_lanelet_by_id(id_suc)
                            last_lanelets.append(successor)
                            # extend
                            vertices_resampled = resample_polyline(successor.center_vertices, step_resample)
                            num_vertices = len(vertices_resampled)
                            di = np.array([[0.0] for _ in range(len(vertices_resampled))])
                            if successor.adj_left in reference_lanelets[i+1]:
                                di = np.array([[1.0] for _ in range(len(vertices_resampled))])
                            elif successor.adj_right in reference_lanelets[i+1]:
                                di = np.array([[-1.0] for _ in range(len(vertices_resampled))])
                            lanelet_leftmost = successor
                            while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                                lanelet_leftmost = self.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                            lw = lanelet_leftmost.distance_line2line(vertices_resampled, line="left")

                            lanelet_rightmost = successor
                            while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                                lanelet_rightmost = self.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                            lr = lanelet_rightmost.distance_line2line(vertices_resampled, line="right")

                            if successor.stop_line is not None:
                                #start = lanelet.stop_line.start
                                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                                ind = np.argmin(np.sqrt(np.sum((vertices_resampled - successor.stop_line.start)**2, axis=1)))
                                sl[ind, 0] = 1.0
                            else:
                                sl = np.array([[0.0] for _ in range(len(vertices_resampled))])
                            path_to_be_concatenated = np.concatenate((vertices_resampled, di, lw, lr, sl), axis=1)
                            reference_segments[i] =  np.concatenate((reference_segments[i], path_to_be_concatenated[1:]), axis=0)
                            reference_lanelets[i].extend([id_suc for _ in range(len(vertices_resampled)-1)])
                            reference_from_to[i].append(np.array([len(reference_segments[i])-len(vertices_resampled)+1, len(reference_segments[i])]))

                            dict_ids_lanelets[id_suc] = i
                            #if successor.adj_left is not None and successor.adj_left_same_direction and successor.adj_left not in dict_ids_lanelets:
                            #    dict_ids_lanelets[successor.adj_left] = i
                            #if successor.adj_right is not None and successor.adj_right_same_direction and successor.adj_right not in dict_ids_lanelets:
                            #    dict_ids_lanelets[successor.adj_right] = i

                        elif id_suc in self.list_ids_lanelets:
                            for j in range(i, len(reference_lanelets)):
                                if reference_lanelets[j][0] == id_suc:
                                    old_ind = j
                                    break
                            if old_ind == i:
                                continue
                            path_to_be_concatenated = reference_segments[old_ind]
                            reference_segments[i] =  np.concatenate((reference_segments[i], path_to_be_concatenated[1:]), axis=0)
                            reference_lanelets[i] = reference_lanelets[i] + reference_lanelets[old_ind]
                            #reference_from_to[i].append(np.array([len(reference_segments[i])-len(vertices_resampled)+1, len(reference_segments[i])]))
                            del reference_lanelets[old_ind]
                            del reference_segments[old_ind]

        principle_reference = [0 for _ in range(len(reference_lanelets))]
        has_goal, has_start = False , False
        for i in reversed(range(len(reference_lanelets))):
            id_lanelets, segment  = reference_lanelets[i], reference_segments[i]
            if id_lanelets[-1] == self.list_ids_lanelets[-1]:
                has_goal = True
                principle_reference[i] = 1
            if id_lanelets[0] == self.list_ids_lanelets[0]:
                has_start = True
            if has_goal and has_start:
                break

        for seg in reference_segments:
            try:
                seg[:, 0] = savgol_filter(seg[:, 0], 25, 5)
                seg[:, 1] = savgol_filter(seg[:, 1], 25, 5)
            except:
                continue
        return reference_segments, reference_lanelets, principle_reference


class RouteCandidateHolder:
    """Class to hold route candidates generated by the route planner"""

    def __init__(self, lanelet_network: LaneletNetwork, planning_problem: PlanningProblem, list_route_candidates: List[List[int]],
                 route_type: RouteType, set_ids_lanelets_permissible: Set):

        self.planning_problem = planning_problem
        self.lanelet_network = lanelet_network

        # create a list of Route objects for all routes found by the route planner which is not empty
        self.list_route_candidates = [Route(lanelet_network, planning_problem, route, route_type, set_ids_lanelets_permissible)
                                      for route in list_route_candidates if route]
        self.num_route_candidates = len(self.list_route_candidates)

        if set_ids_lanelets_permissible is None:
            self.set_ids_lanelets_permissible = {lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets}
        else:
            self.set_ids_lanelets_permissible = set_ids_lanelets_permissible

        self.route_type = route_type

    def retrieve_first_route(self) -> Route:
        return self.list_route_candidates[0]

    def retrieve_best_route_by_orientation(self) -> Union[Route, None]:
        """Retrieves the best route found by some orientation metrics

        If it is the survival scenario, then the first route with idx 0 is returned.
        """
        if not len(self.list_route_candidates):
            return None

        if self.route_type == RouteType.SURVIVAL:
            return self.retrieve_first_route()

        else:
            state_current = self.planning_problem.initial_state
            # sort the lanelets in the scenario based on their orientation difference with the initial state
            list_ids_lanelets_initial_sorted = sort_lanelet_ids_by_orientation(
                self.lanelet_network.find_lanelet_by_position([state_current.position])[0],
                state_current.orientation,
                state_current.position,
                self.lanelet_network
            )
            # sort the lanelets in the scenario based on the goal region metric
            list_ids_lanelets_goal_sorted = sort_lanelet_ids_by_goal(self.lanelet_network, self.planning_problem.goal)

            list_ids_lanelet_goal_candidates = np.array(
                [route_candidate.list_ids_lanelets[-1] for route_candidate in self.list_route_candidates])

            for id_lanelet_goal in list_ids_lanelets_goal_sorted:
                if id_lanelet_goal in list_ids_lanelet_goal_candidates:
                    list_ids_lanelets_initial_candidates = list()
                    for route_candidate in self.list_route_candidates:
                        if route_candidate.list_ids_lanelets[-1] == id_lanelet_goal:
                            list_ids_lanelets_initial_candidates.append(route_candidate.list_ids_lanelets[0])
                        else:
                            list_ids_lanelets_initial_candidates.append(None)
                    list_ids_lanelets_initial_candidates = np.array(list_ids_lanelets_initial_candidates)

                    for initial_lanelet_id in list_ids_lanelets_initial_sorted:
                        if initial_lanelet_id in list_ids_lanelets_initial_candidates:
                            route = self.list_route_candidates[
                                np.where(list_ids_lanelets_initial_candidates == initial_lanelet_id)[0][0]]
                            return route
            return None

    def retrieve_all_routes(self) -> Tuple[List[Route], int]:
        """ Returns the list of Route objects and the total number of routes"""
        return self.list_route_candidates, self.num_route_candidates

    def __repr__(self):
        return f"{len(self.list_route_candidates)} routeCandidates of scenario, " \
               f"planning problem {self.planning_problem.planning_problem_id}"

    def __str__(self):
        return self.__repr__()

class RoutePlanner:
    def __init__(self, lanelet_network: LaneletNetwork,
                 planning_problem: PlanningProblem,
                 reach_goal_state: bool = False):

        self.lanelet_network = lanelet_network
        self.planning_problem = planning_problem
        self.reach_goal_state = reach_goal_state

        # find permissible lanelets
        self.set_ids_lanelets_permissible = {lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets}

        # examine initial and goal lanelet ids
        self.id_lanelets_start = self._retrieve_ids_lanelets_start()

        self.ids_lanelets_goal, self.ids_lanelets_goal_original = self._retrieve_ids_lanelets_goal()

        if self.reach_goal_state and not self.ids_lanelets_goal:
            # if the predecessors of the goal states cannot be reached, fall back to reaching the goal lanelets
            self.reach_goal_state = False
            self.ids_lanelets_goal, self.ids_lanelets_goal_original = self._retrieve_ids_lanelets_goal()

        self._create_lanelet_network_graph()

    def _retrieve_ids_lanelets_start(self):
        """Retrieves the ids of the lanelets in which the initial position is situated"""
        if hasattr(self.planning_problem.initial_state, 'posF'):
            list_ids_lanelets_start = [self.planning_problem.initial_state.posF.ind[-1]]
            self.ids_lanelets_start_overtake = list()

        elif hasattr(self.planning_problem.initial_state, 'position'):
            post_start = self.planning_problem.initial_state.position
            # noinspection PyTypeChecker
            list_ids_lanelets_start = self.lanelet_network.find_lanelet_by_position([post_start])[0]

            list_ids_lanelets_start = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_start))

            # Check if any of the start positions are during an overtake:
            # if the car is not driving in the correct direction for the lanelet,
            # it will also consider routes taking an adjacent lanelet in the opposite direction
            self.ids_lanelets_start_overtake = list()
            if (hasattr(self.planning_problem.initial_state, 'orientation')
                    and not self.planning_problem.initial_state.is_uncertain_orientation):
                orientation = self.planning_problem.initial_state.orientation

                for id_lanelet_start in list_ids_lanelets_start:
                    lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet_start)
                    lanelet_angle = lanelet_orientation_at_position(lanelet, post_start)

                    # Check if the angle difference is larger than 90 degrees
                    if abs(relative_orientation(orientation, lanelet_angle)) > 0.5 * np.pi:
                        if (lanelet.adj_left is not None and not lanelet.adj_left_same_direction
                                and lanelet.adj_left in self.set_ids_lanelets_permissible):
                            self.ids_lanelets_start_overtake.append((id_lanelet_start, lanelet.adj_left))

                        elif (lanelet.adj_right is not None and not lanelet.adj_right_same_direction
                              and lanelet.adj_right in self.set_ids_lanelets_permissible):
                            self.ids_lanelets_start_overtake.append((id_lanelet_start, lanelet.adj_right))
        else:
            raise

        return list_ids_lanelets_start

    def _filter_allowed_lanelet_ids(self, list_ids_lanelets_to_filter: List[int]) \
            -> Generator[int, None, None]:
        """Filters lanelets with the list of ids of forbidden lanelets.

        :param list_ids_lanelets_to_filter: The list of the lanelet ids which should be filtered
        :return: List of desirable lanelets
        """
        for id_lanelet in list_ids_lanelets_to_filter:
            if id_lanelet in self.set_ids_lanelets_permissible:
                yield id_lanelet

    def _retrieve_ids_lanelets_goal(self):
        """Retrieves the ids of the lanelets in which the goal position is situated"""
        list_ids_lanelets_goal = list()
        list_ids_lanelets_goal_original = list()

        if hasattr(self.planning_problem.goal, 'lanelets_of_goal_position'):
            if self.planning_problem.goal.lanelets_of_goal_position is not None:
                # the goals are stored in a dict, one goal can consist of multiple lanelets
                # now we just iterate over the goals and add every ID which we find to
                # the goal_lanelet_ids list
                for list_ids_lanelets_pos_goal in list(self.planning_problem.goal.lanelets_of_goal_position.values()):
                    list_ids_lanelets_goal.extend(list_ids_lanelets_pos_goal)

                list_ids_lanelets_goal = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_goal))

        if list_ids_lanelets_goal:
            self.reach_goal_state = False

        elif hasattr(self.planning_problem.goal, 'state_list'):
            for idx, state in enumerate(self.planning_problem.goal.state_list):
                if hasattr(state, 'posF'):
                    list_ids_lanelets_pos_goal = [state.posF.ind[-1]]
                    if list_ids_lanelets_pos_goal:
                        list_ids_lanelets_goal.extend(list_ids_lanelets_pos_goal)
                if hasattr(state, 'position'):
                    if hasattr(state.position, 'center'):
                        pos_goal = state.position.center

                    else:
                        pos_goal = state.position
                    [list_ids_lanelets_pos_goal] = self.lanelet_network.find_lanelet_by_position([pos_goal])
                    list_ids_lanelets_pos_goal = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_pos_goal))

                    if self.reach_goal_state:
                        # we want to reach the goal states (not just the goal lanelets), here we instead demand
                        # reaching the predecessor lanelets of the goal states
                        list_ids_lanelets_goal_original = list_ids_lanelets_pos_goal.copy()
                        list_ids_lanelets_pos_goal.clear()

                        for id_lanelet_goal in list_ids_lanelets_goal_original:
                            lanelet_goal = self.lanelet_network.find_lanelet_by_id(id_lanelet_goal)
                            # make predecessor as goal
                            list_ids_lanelets_pos_goal.extend(lanelet_goal.predecessor)

                    if list_ids_lanelets_pos_goal:
                        list_ids_lanelets_goal.extend(list_ids_lanelets_pos_goal)


        # remove duplicates and reset to none if no lanelet IDs found
        if list_ids_lanelets_goal:
            # remove duplicates and sort in ascending order
            list_ids_lanelets_goal = sorted(list(dict.fromkeys(list_ids_lanelets_goal)))
        else:
            list_ids_lanelets_goal = None

        return list_ids_lanelets_goal, list_ids_lanelets_goal_original

    def _create_lanelet_network_graph(self):
        """Creates a directed graph of lanelets."""

        if self.ids_lanelets_goal is None:
            # if there is no goal lanelet ids then it is a survival scenario and
            # we do not need to make a graph from the lanelet network
            self.route_type = RouteType.SURVIVAL

        else:
            # construct directed graph
            self.route_type = RouteType.REGULAR
            self.digraph = self._create_graph_from_lanelet_network()

    def _create_graph_from_lanelet_network(self) -> nx.DiGraph:
        """Builds a graph from the lanelet network

        Edges are added from the successor relations between lanelets.

        :return: created graph from lanelet network
        """

        graph = nx.DiGraph()
        nodes = list()
        edges = list()

        for lanelet in self.lanelet_network.lanelets:
            # only accept allowed lanelets
            if lanelet.lanelet_id not in self.set_ids_lanelets_permissible:
                continue

            nodes.append(lanelet.lanelet_id)

            # add edge if succeeding lanelets exist
            for id_successor in lanelet.successor:
                if id_successor not in self.set_ids_lanelets_permissible:
                    continue
                edges.append((lanelet.lanelet_id, id_successor, {'weight': lanelet.distance[-1]}))

            # add edge if left lanelet
            id_adj_left = lanelet.adj_left
            if id_adj_left and lanelet.adj_left_same_direction and id_adj_left in self.set_ids_lanelets_permissible:
                lanelet_adj_left = self.lanelet_network.find_lanelet_by_id(id_adj_left)
                dist2start = np.sqrt(np.sum((self.planning_problem.initial_state.position - lanelet_adj_left.center_vertices[0])**2))

                edges.append((lanelet.lanelet_id, id_adj_left, {'weight': 400.0+dist2start})) # discourage lane change

            # add edge if right lanelet
            id_adj_right = lanelet.adj_right
            if id_adj_right and lanelet.adj_right_same_direction and id_adj_right in self.set_ids_lanelets_permissible:
                lanelet_adj_right = self.lanelet_network.find_lanelet_by_id(id_adj_right)
                dist2start = np.sqrt(np.sum((self.planning_problem.initial_state.position - lanelet_adj_right.center_vertices[0])**2))

                edges.append((lanelet.lanelet_id, id_adj_right, {'weight': 400.0+dist2start})) # discourage lane change

        # Edges in case of overtake during starting state
        for id_start, id_adj in self.ids_lanelets_start_overtake:
            edges.append((id_start, id_adj, {'weight': 1.0}))

        # add all nodes and edges to graph
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def plan_routes(self):
        """Plans routes for every pair of start/goal lanelets.

        If no goal lanelet ID is given then return a survival route.
        :return: list of lanelet ids from start to goal.
        """
        # route is a list that holds lists of lanelet ids from start lanelet to goal lanelet
        list_routes = list()

        # iterate through start lanelet ids
        for id_lanelet_start in self.id_lanelets_start:
            if self.ids_lanelets_goal:
                # iterate through goal lanelet ids
                for id_lanelet_goal in self.ids_lanelets_goal:
                    list_lists_ids_lanelets = list()
                    list_lists_ids_lanelets = self._find_routes_networkx(id_lanelet_start, id_lanelet_goal)

                    if list_lists_ids_lanelets:
                        if self.reach_goal_state:
                            # append the original goal lanelet back to the found route
                            for id_lanelet_goal_original in self.ids_lanelets_goal_original:
                                for list_ids_lanelets in list_lists_ids_lanelets:
                                    list_routes.append(list_ids_lanelets + [id_lanelet_goal_original])

                        else:
                            list_routes.extend(list_lists_ids_lanelets)

            else:
                # no goal lanelet, find survival route
                list_lists_ids_lanelets = self._find_survival_route(id_lanelet_start)
                list_routes.append(list_lists_ids_lanelets)

        return RouteCandidateHolder(self.lanelet_network, self.planning_problem, list_routes,
                                    self.route_type, self.set_ids_lanelets_permissible)
    def plan_alternative_routes(self):
        list_routes = list()
        self.ids_lanelets_alternative_goal = list()
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.successor) == 0 and lanelet.lanelet_id not in self.ids_lanelets_goal:
                self.ids_lanelets_alternative_goal.append(lanelet.lanelet_id)
        for id_lanelet_start in self.id_lanelets_start:
            for id_lanelet_goal in self.ids_lanelets_alternative_goal:
                list_lists_ids_lanelets = list()
                list_lists_ids_lanelets = self._find_routes_networkx(id_lanelet_start, id_lanelet_goal)

                if list_lists_ids_lanelets:
                    if self.reach_goal_state:
                        # append the original goal lanelet back to the found route
                        for id_lanelet_goal_original in self.ids_lanelets_goal_original:
                            for list_ids_lanelets in list_lists_ids_lanelets:
                                list_routes.append(list_ids_lanelets + [id_lanelet_goal_original])

                    else:
                        list_routes.extend(list_lists_ids_lanelets)
        return RouteCandidateHolder(self.lanelet_network, self.planning_problem, list_routes,
                                    self.route_type, self.set_ids_lanelets_permissible)

    def _find_routes_networkx(self, id_lanelet_start: int, id_lanelet_goal: int = None) -> List[List]:
        """Find all shortest paths using networkx module

        This tends to change lane late.
        :param id_lanelet_start: ID of start lanelet
        :param id_lanelet_goal: ID of goal lanelet
        :return: list of lists of lanelet IDs
        """
        list_lanelets = list()

        if id_lanelet_start is None:
            raise

        if id_lanelet_goal is None:
            raise
        try:
            list_lanelets = list(nx.all_shortest_paths(self.digraph, source=id_lanelet_start, target=id_lanelet_goal,
                                                       weight='weight', method='dijkstra'))
        except nx.exception.NetworkXNoPath:
            list_lanelets = list()
        return list_lanelets

    def _find_survival_route(self, id_lanelet_start: int) -> List:
        """Finds a route along the lanelet network for survival scenarios.

        The planner advances in the order of forward, right, left whenever possible.
        Notes:
            - it only considers lanes with same driving direction
            - the priorities of right and left should be swapped for left-hand traffic countries, e.g. UK
            - it goes until the end of the lanelet network or when it is hits itself (like dying in the Snake game)

        :param id_lanelet_start: the initial lanelet where we start from
        :return: route that consists of a list of lanelet IDs
        """
        route = list()
        id_lanelet_current = id_lanelet_start
        lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet_current)

        while id_lanelet_current not in route:
            route.append(lanelet.lanelet_id)

            found_new_lanelet = False
            if lanelet.successor:
                # naively select the first successors
                lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.successor[0])
                found_new_lanelet = True

            if not found_new_lanelet and lanelet.adj_right and lanelet.adj_right_same_direction:
                # try to go right
                lanelet_adj_right = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                if len(lanelet_adj_right.successor):
                    # right lanelet has successor
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                    found_new_lanelet = True

            if not found_new_lanelet and lanelet.adj_left and lanelet.adj_left_same_direction:
                # try to go left
                lanelet_adj_left = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                if len(lanelet_adj_left.successor):
                    # left lanelet has successor
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                    found_new_lanelet = True

            if not found_new_lanelet:
                # no possible route to advance
                break
            else:
                # set lanelet
                id_lanelet_current = lanelet.lanelet_id

        return route

def relative_orientation(angle_1, angle_2):
    """Computes the angle between two angles."""

    phi = (angle_2 - angle_1) % (2 * np.pi)
    if phi > np.pi:
        phi -= (2 * np.pi)

    return phi

def lanelet_orientation_at_position(lanelet: Lanelet, position: np.ndarray):
    """Approximates the lanelet orientation with the two closest point to the given state

    :param lanelet: Lanelet on which the orientation at the given state should be calculated
    :param position: Position where the lanelet's orientation should be calculated
    :return: An orientation in interval [-pi,pi]
    """
    center_vertices = lanelet.center_vertices

    position_diff = []
    for idx in range(len(center_vertices) - 1):
        vertex1 = center_vertices[idx]
        position_diff.append(np.linalg.norm(position - vertex1))

    closest_vertex_index = position_diff.index(min(position_diff))

    vertex1 = center_vertices[closest_vertex_index, :]
    vertex2 = center_vertices[closest_vertex_index + 1, :]
    direction_vector = vertex2 - vertex1
    return np.arctan2(direction_vector[1], direction_vector[0])


def sort_lanelet_ids_by_orientation(list_ids_lanelets: List[int], orientation: float, position: np.ndarray,
                                    lanelet_network: LaneletNetwork) \
        -> List[int]:
    """Returns the lanelets sorted by relative orientation to the given position and orientation."""

    if len(list_ids_lanelets) <= 1:
        return list_ids_lanelets
    else:
        lanelet_id_list = np.array(list_ids_lanelets)

        def get_lanelet_relative_orientation(lanelet_id):
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            lanelet_orientation = lanelet_orientation_at_position(lanelet, position)
            return np.abs(relative_orientation(lanelet_orientation, orientation))

        orientation_differences = np.array(list(map(get_lanelet_relative_orientation, lanelet_id_list)))
        sorted_indices = np.argsort(orientation_differences)
        return list(lanelet_id_list[sorted_indices])
