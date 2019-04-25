import random
import matplotlib.pyplot as plt 
from math import sin, cos, pi 
from matplotlib.patches import Ellipse
import math
import numpy
from collections import defaultdict
from heapq import *
from timeit import default_timer as timer
import argparse
from matplotlib.widgets import Slider, Button, RadioButtons


class Map:
  def __init__(self, obstacles=[], start=None, goal=None, plt=None, xBoundary=(-40,40), yBoundary=(-40,40), robot = [], num_obs = 30, num_lines = 40):
    self.num_obs = num_obs
    self.num_lines = num_lines

    # obstacles in the map
    self.obstacles = obstacles

    # minkoski grown obstacles in map
    self.CSpaceObstacles = []

    # start and goal locations
    self.start = start
    self.goal = goal

    # boundaries of the map
    self.startx, self.endx = xBoundary
    self.starty, self.endy = yBoundary

    # Configure and set the figure for plotting
    self.fig = plt.figure(0)
    ax = self.fig.add_subplot(111, aspect='equal')
    ax.set_xlim(self.startx, self.endx)
    ax.set_ylim(self.starty, self.endy)
    self.ax = ax
    self.increment = 1

    # the robot used to navigate through the map
    self.robot = robot

    # dimension of the ellipse robot in (w,h) form
    self.dimensions = (0,0)

    # edges between free space in the map 
    self.edges = [] 

    # dictionary used to link vertices between C-Layers
    self.last_vertices = {}

    # width/height dimensions of rotated robot used for minkowski sum 
    self.rot_dim = []
    self.last_rot_dim = []

    # orientation variables used to connect the C-Layers together
    self.orientation = 0
    self.last_orientation = 0
    self.first_orientation = 0
    self.solved = False

  def clear(self, addObstacles = True):
    '''
        Generate a new map and run the Highway Roadmap algorithm again
    '''
    self.ax.cla()
    self.CSpaceObstacles = []
    self.edges = [] 
    self.last_vertices = {}
    self.rot_dim = []
    self.last_rot_dim = []
    self.robot = []
    self.solved=False
    self.orientation = self.first_orientation
    self.ax.set_xlim(self.startx, self.endx)
    self.ax.set_ylim(self.starty, self.endy)
    if addObstacles:
        self.obstacles = []
        self.genObstacles(self.num_obs, self.start, self.goal, self.dimensions)
    self.highwayRoadMap()
    self.fig.canvas.draw()

  def genObstacles(self, n, start, goal, dimension):
    '''
        Randomly generate n obstacles that don't intersect with the start and
        end regions
    '''
    for i in range(n):
        obstacle = (random.randint(2,5),random.randint(2,5),random.randint(-40,40),random.randint(-40,40))
        if contains(obstacle, start, dimensions) or contains(obstacle, goal, dimensions):
            i-=1
            continue
        self.addObstacle(obstacle)

  #### Setters for Map instance variables ####

  def setDimensions(self, dim):
    self.dimensions = dim

  def setOrientation(self, ori):
    self.orientation = ori
    self.first_orientation = ori

  def addObstacle(self, obstacle):
    self.obstacles.append(obstacle)
    self.obstacles = sorted(self.obstacles, key=lambda x: x[2])

  def addStart(self, start):
    self.start=start

  def addGoal(self, goal):
    self.goal=goal

  def addRobot(self, robot):
    self.robot.append(robot)

  
  def minkowski(self, a1,b1,a2,b2,na,s1,s2):
    '''
        the minkowski operations described in the paper. First scales down the
        obstacle and the robot to where the robot is a circle instead of an ellipse,
        then gets the radius of the circle (the offset boundary) and takes the
        minkowski sum at this point and scales the obstacle back up. The result is the
        new boundary of the superellipse
    '''
    r = rot(s1)
    rad = min(a2,b2)
    shrunk = numpy.asarray([[rad/a2, 0],[0, rad/b2]])
    t = numpy.dot(r, numpy.dot(shrunk, numpy.transpose(r)))
    norm = (2/na)*numpy.asarray([(cos(s2)**(2-na))/a1, (sin(s2)**(2-na))/b1])
    inv = numpy.linalg.inv(t)
    top = numpy.dot(numpy.linalg.matrix_power(t, -2), norm) 
    bottom = numpy.dot(numpy.linalg.matrix_power(t, -1),norm)
    mult = rad * (top / numpy.linalg.norm(bottom))
    return mult

  def printSuperEllipse(self, ax, a, b, n, in_x=0, in_y=0):
    '''
        print out superellipse obstacle
    '''
    na = 2 / n 
    step = 30
    piece =(pi * 2)/step 
    xp =[];yp =[] 
    bigX=[];bigY=[]
    w, h = self.dimensions
    t = 0
    rot_dim = self.rotate(self.orientation, self.dimensions)
    stretch_x = rot_dim[0]/2
    stretch_y = rot_dim[1]/2
    for t1 in range(step + 1): 
        cx = ((abs((cos(t)))**na)*a+stretch_x) * sgn(cos(t)) + in_x 
        cy = ((abs((sin(t)))**na)*b+stretch_y) * sgn(sin(t)) + in_y 

        x =(abs((cos(t)))**na)*a * sgn(cos(t)) + in_x
        y =(abs((sin(t)))**na)*b * sgn(sin(t)) + in_y
        xp.append(x);yp.append(y) 
        bigX.append(cx);bigY.append(cy) 
        t+= piece 

    # plot obstacle in blue and c-space/minkowski obstacle in black outline
    ax.fill(xp, yp, "b")

  def printRobot(self):
    '''
        print out the robot at each vertex along the path from start to goal
    '''
    for e in self.robot:
        self.ax.add_artist(e)
        e.set_clip_box(self.ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([1,0,1])

  def getDistance(self, p1, p2):
    '''
        Get Euclidean distance between two points
    '''
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

  def getCost(self, start, edges):
    '''
        Get the cost of moving from a start location along the set of vertices 
        contained in edges
    '''
    cost = 0
    for i in range(len(edges)):
        cost += self.getDistance(start, edges[i])
        start = edges[i]
    return cost

  def makeEdge(self, start, edges):
    '''
        Add edge between vertices in free space and associate cost of moving
        from start to the last vertex in edges
    '''
    cost = self.getCost(start, edges)
    self.edges.append([start, edges[0], cost])
    for i in range(1, len(edges)):
        self.edges.append([edges[i-1], edges[i], 0])

  def rotate(self, sigma, tu):
    '''
        Get width and heigh of rotated ellipse for use in minkowski operations.
    '''
    x,y = tu
    o = numpy.radians(sigma)
    a = abs(x * sin(o)) + abs(y * cos(o))
    b = abs(x * cos(o)) + abs(y * sin(o))
    return (b,a)

  def minkowskiOperations(self):
    '''
        Transform obstacles to C-Space obstacles
    '''
    self.CSpaceObstacles = []
    self.rot_dim = self.rotate(self.orientation, self.dimensions)
    for ob in self.obstacles:
        self.CSpaceObstacles.append((ob[0]+(self.rot_dim[0]/2), ob[1]+(self.rot_dim[1]/2), ob[2], ob[3]))

  def connectVerticesBetweenCLayer(self, decomp):
    '''
        Connect nearby vertices between adjacent C-Layers to account for
        changing orientation of the robot
    '''
    diff = abs((self.last_rot_dim[0]-self.rot_dim[0])/4)
    for line in decomp:
        for segment in line:
            for i in range(-1, 2):
                if self.last_vertices.get((segment[2]+diff*i, segment[3]),False):
                    start = (segment[2]+diff*i, segment[3], self.last_orientation)
                    edges = [(segment[2], segment[3], self.orientation)]
                    self.makeEdge(start, edges)

  def connectVerticesWithinCLayer(self, decomp):
    '''
        Connect vertices in free space within a single C-Layer. After this runs
        the free space within the current C-Layer will be completely connected.
    '''
    currLine = 0
    for currLine in range(0, len(decomp)-1):
        if len(decomp[currLine][0])==5:
            decomp[currLine]+=decomp[currLine-1]
        for segment in decomp[currLine]:
            for nextSegment in decomp[currLine+1]:
                if segment[1] <= nextSegment[0] or segment[0] >= nextSegment[1]:
                    continue
                if (segment[1] < nextSegment[1] and segment[0] < nextSegment[0]):
                    edges = [(segment[1]-.1, segment[3], self.orientation), (segment[1]-.1, nextSegment[3], self.orientation), (nextSegment[2], nextSegment[3], self.orientation)]
                    self.makeEdge((segment[2],segment[3], self.orientation), edges)
                elif (segment[0] > nextSegment[0] and segment[1] > nextSegment[1]):
                    edges = [(segment[0]+.1, segment[3], self.orientation), (segment[0]+.1, nextSegment[3], self.orientation), (nextSegment[2], nextSegment[3], self.orientation)]
                    self.makeEdge((segment[2],segment[3], self.orientation), edges)
                elif segment[1] < nextSegment[1] or segment[0] > nextSegment[0]:
                    edges = [(segment[2], nextSegment[3], self.orientation), (nextSegment[2], nextSegment[3], self.orientation)]
                    self.makeEdge((segment[2],segment[3], self.orientation), edges)
                else:
                    edges = [(nextSegment[2], segment[3], self.orientation), (nextSegment[2], nextSegment[3], self.orientation)]
                    self.makeEdge((segment[2],segment[3], self.orientation), edges)

  def getBounds(self, coor):
    '''
        Get the left and right bounds of the start and goal location to aid in
        cell decomposition.
    '''
    curr =  [ob for ob in self.CSpaceObstacles if coor[1] >= (ob[3] - ob[1])  and coor[1] <= (ob[3] + ob[1])]
    left = self.startx
    right = self.endx
    for ob in curr:
        leftSide = ob[2]-ob[0]
        rightSide = ob[2]+ob[0]
        if contains(ob, coor, self.dimensions, True):
            return None
        if rightSide < coor[0] :
            left = rightSide
        elif leftSide > coor[0]:
            right = leftSide
            break
    return (coor[0], coor[1], coor[2], left, right)

  def cellDecomposition(self, sweepLines):
    '''
        Characterize the free space of a given C-Layer by sending out a number
        of horizontal sweep lines and save line segments in free space. The
        midpoint of these line segments serve as the vertices for path planning
        as they are the points farthest away from obstacles.
    '''
    ret = []
    step = int((self.endy-self.starty)/sweepLines)
    if step == 0:
        step = 1

    s_ins = g_ins = False
    s_bounds = self.getBounds(self.start)
    e_bounds = self.getBounds(self.goal)
    if s_bounds == None:
        s_ins = True
    if e_bounds == None:
        g_ins = True

    for i in range(self.starty, self.endy+1, step):
        curr =  [ob for ob in self.CSpaceObstacles if i >= (ob[3] - ob[1])  and i <= (ob[3] + ob[1])]
        ls = []
        left = self.startx
        last = None
        for ob in curr:
            right = ob[2]-ob[0]-.1
            if(left > right):
                left = ob[2]+ob[0]+.1
                continue
            ls.append([left,right, (left+right)/2, i])
            left = ob[2]+ob[0]+.1
            last = right
        right = self.endx
        ls.append([left,right, (left+right)/2, i])
        ret.append(ls)
        if not s_ins and self.start[1] < i+step:
            ret.append([[s_bounds[3],s_bounds[4], s_bounds[0], s_bounds[1], True]])
            s_ins=True
        if not g_ins and self.goal[1] < i+step:
            ret.append([[e_bounds[3],e_bounds[4], e_bounds[0], e_bounds[1], True]])
            g_ins=True
    return ret

  def printMap(self, shortest_path=None):
    '''
        print out the state of the map after planning has occured
    '''
    # print out the obstacles
    for ob in self.obstacles:
        self.printSuperEllipse(self.ax, ob[0], ob[1], 20, ob[2], ob[3])

    #print start and goal locations
    self.ax.plot(self.start[0], self.start[1], color='green', marker='o', markersize=4)
    self.ax.plot(self.goal[0], self.goal[1], color='red', marker='o', markersize=4)
    
    # print out the shortest path
    if shortest_path != None and shortest_path != float("inf"):
        self.solved = True
        xs = [i[0] for i in shortest_path]
        ys = [i[1] for i in shortest_path]
        self.ax.plot(xs, ys, color='red', linewidth=1, markersize=1)
        for coor in shortest_path:
            e = Ellipse(xy=coor[:2], width=self.dimensions[0], height=self.dimensions[1], angle=coor[2])
            ma.addRobot(e)
    else:
        self.ax.text(-23, 43, r'No solution possible!', fontsize=15, color='r')

    # print out every orientation of the robot from start to goal locations
    self.printRobot()
  
  def getOrientations(self, start_orientation):
    '''
        Discretize orientiations. Length of orientations list is the number of
        C-Layers that will be generated by the planner.
    '''
    orientations = []
    for ori in range(start_orientation, start_orientation+181, 6):
        orientations.append(ori)
    return orientations

  def highwayRoadMap(self):
    '''
        The highway roadmap algorithm described in the WAFR paper this project
        is inspired by. It works by building up connected C-Layers (where a
        C-Layer is a connected graph of the free space in the map at a certain
        orientation) and then connecting adjacent C-Layers to make a fully
        connected graph. Dijkstra's algorithm is then run on this graph.
    '''

    if self.solved:
        return
    start = timer()
    # Discretize the orientations
    discretized_orientiations = self.getOrientations(self.orientation)

    # Build up a C-Layer for each orientation of the robot
    for ori in discretized_orientiations:
        self.orientation = ori

        # Transform the obstacles to C-Space obstacles
        self.minkowskiOperations()

        # perform cell decomposition and connect the free space vertices
        decomp = self.cellDecomposition(self.num_lines)
        self.connectVerticesWithinCLayer(decomp)

        # Connect vertices between adjacent C-Layers
        if len(self.last_vertices) != 0:
            self.connectVerticesBetweenCLayer(decomp)

        self.last_rot_dim = self.rot_dim
        self.last_vertices = {}
        self.last_orientation = self.orientation
        for line in decomp:
            for segment in line:
                self.last_vertices[tuple(segment[2:4])] = True
    
    # find shortest path from start to goal location across all c-layers
    shortest_path = dijkstra(self.edges, tuple(self.start[:3]), tuple(self.goal[:3]))
    end = timer()
    print("Solution found in %s seconds" % (end - start,))
    self.printMap(shortest_path)
    self.fig.canvas.draw()

def rot(sigma):
    return numpy.asarray([[math.cos(sigma), -math.sin(sigma)],[math.sin(sigma), math.cos(sigma)]])

def sgn(x): 
    return ((x>0)-(x<0))*1 

def contains(ob, coor, dimensions, inCSpace = False):
    '''
        Helper function to ensure randomly generated obstacles don't overlap
        with start or goal location.
    '''
    ob_lside = ob[2]-ob[0]
    ob_rside = ob[2]+ob[0]
    ob_top = ob[3]+ob[1]
    ob_bottom = ob[3]-ob[1]
    ro_lside = coor[0]
    ro_rside = coor[0]
    ro_top = coor[1]
    ro_bottom = coor[1]
    if not inCSpace:
        ro_lside -= (dimensions[0]/2)
        ro_rside += (dimensions[0]/2)
        ro_top += (dimensions[1]/2)
        ro_bottom -= (dimensions[1]/2)
    return not (ro_bottom > ob_top or ob_bottom > ro_top or ro_rside < ob_lside or ob_rside < ro_lside)

def cleanPath(tu):
    '''
        Helper function for dijkstra to return a path of coordinates from start
        to goal
    '''
    cost, path = tu
    ret = []
    while path != ():
        ret.append(path[0])
        path = path[1]
    ret.reverse()
    return ret

def dijkstra(edges, f, t):
    '''
        Version of dijkstra's algorithm using a min heap. Used to navigate from
        start to goal using edges built up via cell decomposition/vertex
        generation.
    '''
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))
        g[r].append((c,l))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: 
                return cleanPath((cost, path))

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")

class Index(object):
    def __init__(self, lines, ma):
        self.lines = lines
        self.ma = ma

    def prev(self, event):
        self.ma.clear()

    def next(self, event):
        self.ma.clear(False)

def update(val):
    lines = sl_lines.val
    obs = sl_obs.val
    ma.num_obs = int(obs)
    ma.num_lines = int(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-sl",help="number of sweep lines to send across",default=30)
    parser.add_argument("-ob",help="number of random obstacles to generate",default=30)

    args = vars(parser.parse_args())

    # define the starting orientation of the robot
    orientation = 0

    # define the number of sweep lines to use in the cell division step
    lines = int(args['sl'])

    # define the start and goal location
    start = [-38,38, orientation]
    goal = [38,-38, orientation]


    # make a map, set the orientation and dimension of the robot
    ma = Map(plt=plt)
    dimensions=(2,6)
    ma.setOrientation(orientation)
    ma.setDimensions(dimensions)

    # Create the buttons and sliders for the GUI
    plt.subplots_adjust(left=0.1, bottom=0.2)
    callback = Index(lines, ma)
    axprev = plt.axes([0.7, 0.001, 0.1, 0.05])
    bprev = Button(axprev, 'New Map')
    bprev.on_clicked(callback.prev)
    axsolve = plt.axes([0.81, 0.001, 0.1, 0.05])
    bsolve = Button(axsolve, 'Solve')
    bsolve.on_clicked(callback.next)
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.001, 0.3, 0.03], facecolor=axcolor)
    sl_obs = Slider(axfreq, '# of obstacles', 1, 50, valinit=30, valstep=1)
    axfreq = plt.axes([0.25, 0.05, 0.3, 0.03], facecolor=axcolor)
    sl_lines = Slider(axfreq, '# of sweeplines', 1, 80, valinit=40, valstep=1)
    sl_lines.on_changed(update)
    sl_obs.on_changed(update)

    # make a certain number of random obstacles to navigate through
    ma.genObstacles(int(args['ob']), start, goal, dimensions)

    # add the start and end location
    ma.addStart(start)
    ma.addGoal(goal)
    ma.highwayRoadMap()
    plt.show()