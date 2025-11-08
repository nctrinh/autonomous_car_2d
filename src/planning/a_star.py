import numpy as np 
import heapq
import time
from typing import List, Tuple, Optional
import sys
from pathlib import Path as pathlib_Path

sys.path.append(str(pathlib_Path(__file__).resolve().parents[2]))

from src.planning.base_planner import BasePlanner, PathPoint, Path
from src.core.map import Map2D


class AStarPlanner(BasePlanner):
    def __init__(self, map_env: Map2D, grid_resolution: float = 0.5):
        self.map_env = map_env
        self.grid_resolution = grid_resolution

        self.grid_width = int(np.ceil(self.map_env.width / self.grid_resolution))
        self.grid_height = int(np.ceil(self.map_env.height / self.grid_resolution))

    def world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        grid_x = int(np.floor(x / self.grid_resolution))
        grid_y = int(np.floor(y / self.grid_resolution))
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x: float, grid_y: float) -> Tuple[float, float]:
        x = (grid_x + 0.5) * self.grid_resolution
        y = (grid_y + 0.5) * self.grid_resolution
        return (x, y)
    
    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float],
             max_iterations: int = 100000,
             heuristic_weight: float = 1.0,
             safety_margin: float = 0.5) -> Optional[Path]:
        start_time = time.time()
        
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        if not self.is_valid_position(start[0], start[1], safety_margin):
            print("A*: Start position is invalid!")
            return None
        
        if not self.is_valid_position(goal[0], goal[1], safety_margin):
            print("A*: Goal position is invalid!")
            return None
        
        open_set = []
        heapq.heappush(open_set, (0, 0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        closed_set = set()
        counter = 0

        self.iterations = 0

        while open_set and self.iterations < max_iterations:
            self.iterations += 1

            _, _, current = heapq.heappop(open_set)

            if current == goal_grid:
                path = self.reconstruct_path(came_from, current)
                self.planning_time = time.time() - start_time
                self.path = path
                
                print(f"A*: Path found! Length: {path.length:.2f}m, "
                      f"Time: {self.planning_time:.3f}s, Iterations: {self.iterations}")
                return path

            closed_set.add(current)

            for neighbor in self._get_grid_neighbors(current, safety_margin):
                if neighbor in closed_set:
                    continue

                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(dx**2 + dy**2) * self.grid_resolution
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    h = self.heuristic(neighbor, goal_grid) * self.grid_resolution
                    f_score[neighbor] = tentative_g + heuristic_weight * h
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        self.planning_time = time.time() - start_time
        print(f"A*: No path found after {self.iterations} iterations!")
        return None

    def _get_grid_neighbors(self, node: Tuple[int, int], 
                           safety_margin: float = 0.5) -> List[Tuple[int, int]]:
        """Get valid neighbors in grid space (8-connectivity)."""
        x, y = node
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue

                world_x, world_y = self.grid_to_world(nx, ny)
                if self.is_valid_position(world_x, world_y, safety_margin):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _reconstruct_path_grid(self, came_from: dict, 
                               current: Tuple[int, int],
                               start: Tuple[int, int]) -> Path:
        """Reconstruct path from grid coordinates to world coordinates."""
        path_points = []
        
        grid_path = [current]
        while current in came_from:
            current = came_from[current]
            grid_path.append(current)
        
        grid_path.reverse()
        
        for grid_x, grid_y in grid_path:
            x, y = self.grid_to_world(grid_x, grid_y)
            path_points.append(PathPoint(x, y))
        
        path = Path(path_points)
        
        path = self._smooth_path(path)
        
        return path
    def _smooth_path(self, path: Path, max_iterations: int = 100) -> Path:
        """
        Smooth path by removing unnecessary waypoints.
        Uses line-of-sight shortcutting.
        """
        if len(path.points) <= 2:
            return path
        
        smoothed = [path.points[0]]
        current_idx = 0
        
        while current_idx < len(path.points) - 1:
            # Try to connect to furthest visible point
            furthest_idx = current_idx + 1
            
            for i in range(len(path.points) - 1, current_idx, -1):
                p1 = path.points[current_idx].to_tuple()
                p2 = path.points[i].to_tuple()
                
                if self.is_path_valid(p1, p2, num_samples=20):
                    furthest_idx = i
                    break
            
            current_idx = furthest_idx
            smoothed.append(path.points[current_idx])
        
        return Path(smoothed)
    
    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.sqrt(dx**2 + dy**2)

if __name__ == "__main__":
    from src.core.map import Map2D, CircleObstacle, RectangleObstacle
    
    map_env = Map2D(width=100, height=100)
    map_env.add_obstacle(CircleObstacle(x=30, y=30, radius=8))
    map_env.add_obstacle(RectangleObstacle(x=60, y=50, width=15, height=30))
    map_env.add_obstacle(CircleObstacle(x=70, y=70, radius=6))
    
    planner = AStarPlanner(map_env, grid_resolution=0.5)
    
    start = (10, 10)
    goal = (90, 90)
    
    print(f"Planning from {start} to {goal}")
    print(f"Grid size: {planner.grid_width} x {planner.grid_height}")
    
    path = planner.plan(start, goal, heuristic_weight=1.5, safety_margin=1.0, max_iterations=1000)
    
    if path:
        print(f"\nPath found!")
        print(f"  Length: {path.length:.2f} meters")
        print(f"  Waypoints: {len(path.points)}")
        print(f"  Planning time: {planner.planning_time:.3f} seconds")
        print(f"  Iterations: {planner.iterations}")
        
        # Print first and last few waypoints
        print("\nFirst 3 waypoints:")
        for i, p in enumerate(path.points[:3]):
            print(f"  {i}: {p}")
        
        print("\nLast 3 waypoints:")
        for i, p in enumerate(path.points[-3:], len(path.points) - 3):
            print(f"  {i}: {p}")
    else:
        print("No path found!")
    