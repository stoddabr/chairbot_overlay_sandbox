import cv2
import numpy as np
import math 

def _find_chairbots(image):
  """ returns the chairbot location and orientation in an image """
  # detectMarkers returns: (corners, ids, rejectedImgPoints)

  dictionary = cv2.aruco.getPredefinedDictionary( cv2.aruco.DICT_4X4_50 )
  corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
  found_chairbots = []
  # Checks all fiducials in dictionary
  if len(corners) > 0:
    # fids corners, findex fiducial index
    for (fids, index) in zip(corners, ids):
      for corner in fids: # pt => point number
        try:
          fid = int(index[0]) # defined fiducial id number
          # exclude fiducial ids outside of expected range
          if (fid >= 0 and fid <= 5):  # chairbot max number is 5
            # ll contains (x, y) coordinate of the middle of fiducial
            midcords = (corner[0] + corner[1] +corner[2] +corner[3]) \
                /4 # average sum of 4 corners
            # calculate angle from origin to fiducial center
            # average of the top two fiducial corners
            topcords = (corner[0] + corner[3]) / 2
            # average of the bottome two fiducial corners
            botcords = (corner[1] + corner[2]) / 2
            # Difference between top and bottom
            ydeltacords = topcords - botcords
            # Tangent of the y and the x
            theta = math.atan2(ydeltacords[1], ydeltacords[0])
            # Changes theta from radians to positive degrees (0 to 360 rotating counter-clockwise)
            degree = theta * (180 / math.pi) + 180
            found_chairbots.append([midcords, degree ])
        except IndexError:
          print('IndexError thrown and passed while looping through aruco markers')
          pass
  return found_chairbots

def process_robots(found_bots):
  all_bots = []
  for bot in found_bots:
    x = bot[0][0]
    y = bot[0][1]
    angle = bot[1]
    # round real world angle to 0, 45, 90, ... 360
    roundedangle = int(angle/45)
    correctangle = 45*roundedangle
    currentbot = (x,y,correctangle)
    all_bots.append(currentbot)

  return all_bots

def scale_grid(all_bots, image):
  height, width, _ = image.shape
  # Round the height and width values of the image to nearest tenth
  heightR = int(math.ceil(height / 10.0)) * 10
  widthR = int(math.ceil(width / 10.0)) * 10
  # Scale the height and width values down by a factor of 10 for path planning grid
  heightS = int(heightR/10)
  widthS = int(widthR/10)
  smallgrid = np.zeros((heightS, widthS))
  
  # Scale robot locations by same factor and put them into new list
  scaledbots = []
  for bot in all_bots:
    x = int(bot[0]/10)
    y = int(bot[1]/10)
    angle = bot[2]
    newbot = (x,y,angle)
    scaledbots.append(newbot)

  # Pixel size of robot locations on scalegrid (with buffer included)
  robotsize = 5
  for i in range(robotsize):
    for j in range(robotsize):
      for bot in scaledbots:
        smallgrid[i+bot[1],j+bot[0]] = 255

  return(smallgrid, scaledbots)
# This code looks at angle as well
# The goal of this is to rectify the model used for path planning with how the robot actually moves
# Now position will consist of three dimensions: [x,y,angle]
# Assume an 8-connected grid

class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):               #<-- added a hash method
        return hash(self.position)

def get_adjacent_squares(position):
    """ based on the nonholonomic robot motion return adjacent squares based on position """

    x,y,theta = position
    neighbors = []
    # case 1: turn in place
    for angle in range(0,361, 45):  # loop through all possible angles in 8-grid
        neighbors.append((x, y, angle))
    
    # case 2: move forward
    # 0 is left 90 is up 180 is right and 270 is down 
    # Assuming positive is 1
    forward_delta = {
        0: [-1,0],
        45: [-1,1],
        90: [0,1],
        135: [1,1],
        180: [1,0],
        225: [1,-1],
        270: [0,-1],
        315: [-1,-1],
        360: [-1,0]
    }
    dx, dy = forward_delta[theta]
    neighbors.append((x+dx, y+dy, theta))

    return neighbors

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = set()                # <-- closed_list must be a set

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.add(current_node)     # <-- change append to add

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
              #Only append to path if angle changes in next node
              #Set next node as white pixel to visualize

                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path # TODO return the direction of the first node

        # Generate children
        children = []
        #return get_adjacent_squares(current_node.position)
        for node_position in get_adjacent_squares(current_node.position): # Adjacent squares -- 

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:              # <-- remove inner loop so continue takes you to the end of the outer loop
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def rescale_paths(plannedpaths):
  rescaledpath_multi = []
  for robot in plannedpaths:
    rescaledpath = []
    for path in robot:
      x = path[0]
      y = path[1]
      angle = path[2]
      x = x*10
      y = y*10
      botpath = (x,y,angle)
      rescaledpath.append(botpath)
    rescaledpath_multi.append(rescaledpath)
  return rescaledpath_multi

def simplify_path(rescaledpath):
  simplepath_multi = []
  for path in rescaledpath:
    compare = path[0]
    simplepath = [path[0]]
    for coord in path:
      if (coord[2] != compare[2]):
        compare = coord
        simplepath.append(compare)
    simplepath_multi.append(simplepath)
  return simplepath_multi

def main():
  image = cv2.imread('images/chairbot_noAR_4.png')
  all_robots = process_robots(_find_chairbots(image))
  scaledgrid, scaledbots = scale_grid(all_robots, image)

  plannedpaths = []
  for bot in scaledbots:
    x = bot[0]
    y = bot[1]
    angle = bot[2]
    start = (x,y,angle)
    end = (0,0,0)
    botpath = astar(scaledgrid, start, end)
    plannedpaths.append(botpath)

  rescaledpath = rescale_paths(plannedpaths)
  simplepath = simplify_path(rescaledpath)
  print(simplepath)
  return(simplepath)

main()
