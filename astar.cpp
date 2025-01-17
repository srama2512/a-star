#include <queue>
#include <limits>
#include <cmath>

// represents a single pixel
class Node {
  public:
    int idx;     // index in the flattened grid
    float cost;  // cost of traversing this pixel

    Node(int i, float c) : idx(i),cost(c) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

bool operator==(const Node &n1, const Node &n2) {
  return n1.idx == n2.idx;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}

// L_2 norm (euclidean distance)
float l2_norm(int i0, int j0, int i1, int j1) {
  return sqrt(pow(i0 - i1, 2) + pow(j0 - j1, 2));
}

// obmap:          flattened h x w grid obstacle map, 1 for obstacle, 0 for free
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
extern "C" bool astar(
      const float* obmap, const int h, const int w,
      const int start, const int goal, bool diag_ok,
      int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);
  Node goal_node(goal, 0.);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];
  float nbrs_costs [8] = {sqrt(2), 1, sqrt(2), 1, 1, sqrt(2), 1, sqrt(2)};

  bool solution_found = false;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur == goal_node) {
      solution_found = true;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      // if the neighbor is within bounds and is not an obstacle
      if (nbrs[i] >= 0 && obmap[nbrs[i]] == 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          heuristic_cost = l2_norm(nbrs[i] / w, nbrs[i] % w,
                                   goal    / w, goal    % w);

          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  delete[] costs;
  delete[] nbrs;

  return solution_found;
}


// This algorithm uses a weighted graph. The distance to the neighbor is also
// determined by the proximity of the neighbor to an obstacle. The closer the
// neighbor is to an obstacle, larger the weight.
// obmap:          flattened h x w grid obstacle map, 1 for obstacle, 0 for free
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
extern "C" bool weighted_astar(
      const float* obmap, const int h, const int w,
      const int start, const int goal, bool diag_ok,
      const float wscale, const int niters,
      int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);
  Node goal_node(goal, 0.);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];
  float nbrs_costs [8] = {sqrt(2), 1, sqrt(2), 1, 1, sqrt(2), 1, sqrt(2)};

  // Assign weights based on proximity to obstacles.
  // Obstacles get a weight of wscale ** (niters + 1).
  // For any free node, the weight assigned is:
  //          max(max(neighbor weights) / wscale, 1.0)
  // This assignment is done via multiple iterations.
  // Create a new array to store these weights
  float* old_prox_wts = new float[h * w];
  float* prox_wts = new float[h * w];
  // Assign initial weights.
  for (int i = 0; i < h * w; ++i) {
    old_prox_wts[i] = obmap[i] == 0 ? 1.0 : pow(wscale, niters+1);
  }
  for (int j = 0; j < niters; ++j) {
    // Weight assignment
    for (int i = 0; i < h * w; ++i) {
      if (obmap[i] == 0) {
        int row = i / w;
        int col = i % w;
        // check bounds and find up to eight neighbors: top to bottom, left to right
        nbrs[0] = (diag_ok && row > 0 && col > 0)          ? i - w - 1   : -1;
        nbrs[1] = (row > 0)                                ? i - w       : -1;
        nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? i - w + 1   : -1;
        nbrs[3] = (col > 0)                                ? i - 1       : -1;
        nbrs[4] = (col + 1 < w)                            ? i + 1       : -1;
        nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? i + w - 1   : -1;
        nbrs[6] = (row + 1 < h)                            ? i + w       : -1;
        nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? i + w + 1   : -1;
        // compute maximum weights across neighbors
        float max_wt = 0.0;
        for (int k = 0; k < 8; ++k) {
          if (nbrs[k] >= 0) {
            max_wt = fmax(max_wt, old_prox_wts[nbrs[k]]);
          }
        }
        max_wt = fmax(max_wt / wscale, 1.0);
        prox_wts[i] = max_wt;
      }
      else {
        prox_wts[i] = old_prox_wts[i];
      }
    }
    // Copy over the weights to old_prox_wts
    for (int i = 0; i < h * w; ++i) {
      old_prox_wts[i] = prox_wts[i];
    }
  }

  bool solution_found = false;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur == goal_node) {
      solution_found = true;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      // if the neighbor is within bounds and is not an obstacle
      if (nbrs[i] >= 0 && obmap[nbrs[i]] == 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i] * prox_wts[nbrs[i]];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          heuristic_cost = l2_norm(nbrs[i] / w, nbrs[i] % w,
                                   goal    / w, goal    % w);

          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  delete[] costs;
  delete[] nbrs;
  delete[] prox_wts;
  delete[] old_prox_wts;

  return solution_found;
}


// obmap:          flattened h x w grid obstacle map, 1 for obstacle, 0 for free
// gmap:           flattened h x w grid goal map, 1 for goal, 0 for non-goal
// h, w:           height and width of grid
// start:          index of start location in flattened grid
// goals:          array of indices of a subset of goals from gmap
// n_goals:        the number of goals
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
extern "C" int multi_goal_astar(
      const float* obmap, const float* gmap, const int h, const int w,
      const int start, const int* goals, const int n_goals, bool diag_ok,
      int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];
  float nbrs_costs [8] = {sqrt(2), 1, sqrt(2), 1, 1, sqrt(2), 1, sqrt(2)};

  int solution_idx = -1;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (gmap[cur.idx] == 1) {
      solution_idx = cur.idx;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    float curr_heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      // if the neighbor is within bounds and is not an obstacle
      if (nbrs[i] >= 0 && obmap[nbrs[i]] == 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goals based on legal moves
          heuristic_cost = l2_norm(nbrs[i] / w,  nbrs[i] % w,
                                  goals[0] / w, goals[0] % w);
          for (int j = 1; j < n_goals; ++j){
            curr_heuristic_cost = l2_norm(nbrs[i] / w,  nbrs[i] % w,
                                         goals[j] / w, goals[j] % w);
            if (curr_heuristic_cost < heuristic_cost){
              heuristic_cost = curr_heuristic_cost;
            }
          }
          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  delete[] costs;
  delete[] nbrs;

  return solution_idx;
}



// This algorithm combines multi_goal_astar and weighted_astar.
// obmap:          flattened h x w grid obstacle map, 1 for obstacle, 0 for free
// gmap:           flattened h x w grid goal map, 1 for goal, 0 for non-goal
// h, w:           height and width of grid
// start:          index of start location in flattened grid
// goals:          array of indices of a subset of goals from gmap
// n_goals:        the number of goals
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
extern "C" int multi_goal_weighted_astar(
      const float* obmap, const float* gmap, const int h, const int w,
      const int start, const int* goals, const int n_goals, bool diag_ok,
      const float wscale, const int niters,
      int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];
  float nbrs_costs [8] = {sqrt(2), 1, sqrt(2), 1, 1, sqrt(2), 1, sqrt(2)};

  // Assign weights based on proximity to obstacles.
  // Obstacles get a weight of wscale ** (niters + 1).
  // For any free node, the weight assigned is:
  //          max(max(neighbor weights) / wscale, 1.0)
  // This assignment is done via multiple iterations.
  // Create a new array to store these weights
  float* old_prox_wts = new float[h * w];
  float* prox_wts = new float[h * w];
  // Assign initial weights.
  for (int i = 0; i < h * w; ++i) {
    old_prox_wts[i] = obmap[i] == 0 ? 1.0 : pow(wscale, niters+1);
  }
  for (int j = 0; j < niters; ++j) {
    // Weight assignment
    for (int i = 0; i < h * w; ++i) {
      if (obmap[i] == 0) {
        int row = i / w;
        int col = i % w;
        // check bounds and find up to eight neighbors: top to bottom, left to right
        nbrs[0] = (diag_ok && row > 0 && col > 0)          ? i - w - 1   : -1;
        nbrs[1] = (row > 0)                                ? i - w       : -1;
        nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? i - w + 1   : -1;
        nbrs[3] = (col > 0)                                ? i - 1       : -1;
        nbrs[4] = (col + 1 < w)                            ? i + 1       : -1;
        nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? i + w - 1   : -1;
        nbrs[6] = (row + 1 < h)                            ? i + w       : -1;
        nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? i + w + 1   : -1;
        // compute maximum weights across neighbors
        float max_wt = 0.0;
        for (int k = 0; k < 8; ++k) {
          if (nbrs[k] >= 0) {
            max_wt = fmax(max_wt, old_prox_wts[nbrs[k]]);
          }
        }
        max_wt = fmax(max_wt / wscale, 1.0);
        prox_wts[i] = max_wt;
      }
      else {
        prox_wts[i] = old_prox_wts[i];
      }
    }
    // Copy over the weights to old_prox_wts
    for (int i = 0; i < h * w; ++i) {
      old_prox_wts[i] = prox_wts[i];
    }
  }

  int solution_idx = -1;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (gmap[cur.idx] == 1) {
      solution_idx = cur.idx;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    float curr_heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      // if the neighbor is within bounds and is not an obstacle
      if (nbrs[i] >= 0 && obmap[nbrs[i]] == 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i] * prox_wts[nbrs[i]];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goals based on legal moves
          heuristic_cost = l2_norm(nbrs[i] / w,  nbrs[i] % w,
                                  goals[0] / w, goals[0] % w);
          for (int j = 1; j < n_goals; ++j){
            curr_heuristic_cost = l2_norm(nbrs[i] / w,  nbrs[i] % w,
                                         goals[j] / w, goals[j] % w);
            if (curr_heuristic_cost < heuristic_cost){
              heuristic_cost = curr_heuristic_cost;
            }
          }
          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  delete[] costs;
  delete[] nbrs;
  delete[] prox_wts;
  delete[] old_prox_wts;

  return solution_idx;
}