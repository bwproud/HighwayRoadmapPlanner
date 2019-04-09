# Python program to implement  
# Superellipse 
import random
class Map:
  def __init__(self, obstacles=[], start=None, goal=None, plt=None, xBoundary=(-40,40), yBoundary=(-40,40), robot = []):
    self.obstacles = obstacles
    self.start = start
    self.goal = goal
    self.startx, self.endx = xBoundary
    self.starty, self.endy = yBoundary
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(self.startx, self.endx)
    ax.set_ylim(self.starty, self.endy)
    self.ax = ax
    self.robot = robot 

  def printSuperEllipse(self, ax, a, b, n, in_x=0, in_y=0):
    na = 2 / n 
    # defining the accuracy 
    step = 30 
    piece =(pi * 2)/step 
    xp =[];yp =[] 
    t = 0
    for t1 in range(step + 1): 
        x =(abs((cos(t)))**na)*a * sgn(cos(t)) + in_x
        y =(abs((sin(t)))**na)*b * sgn(sin(t)) + in_y
        xp.append(x);yp.append(y) 
        t+= piece 
    ax.fill(xp, yp, "b")
    ax.plot(xp, yp, "r")
    ax.plot(self.start[0], self.start[1], color='green', marker='o', markersize=4)
    ax.plot(self.goal[0], self.goal[1], color='red', marker='o', markersize=4)
    self.printRobot()

  def addObstacle(self, obstacle):
    self.obstacles.append(obstacle)
    self.obstacles = sorted(self.obstacles, key=lambda x: x[2])

  def addStart(self, start):
    self.start=start

  def addGoal(self, goal):
    self.goal=goal

  def addRobot(self, robot):
    self.robot.append(robot)

  def printRobot(self):
    for e in self.robot:
        self.ax.add_artist(e)
        e.set_clip_box(self.ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([1,0,1])

  def inellipse(self, x,y,a,b,h,k):
    return (x-h)**2/a**2 + (y-k)**2/b**2

  def connectVerticesWithinCLayer(self, decomp):
    currLine = 0
    for currLine in range(0, len(decomp)-1):
        if len(decomp[currLine][0])==5:
            decomp[currLine]+=decomp[currLine-1]
        for segment in decomp[currLine]:
            for nextSegment in decomp[currLine+1]:
                if segment[1] <= nextSegment[0] or segment[0] >= nextSegment[1]:
                    continue
                if (segment[1] < nextSegment[1] and segment[0] < nextSegment[0]):
                    self.ax.plot([segment[2], segment[1]-.1], [segment[3], segment[3]], color='black')
                    self.ax.plot([segment[1]-.1, segment[1]-.1], [segment[3], nextSegment[3]], color='black')
                    self.ax.plot([segment[1]-.1, nextSegment[2]], [nextSegment[3], nextSegment[3]], color='black')
                elif (segment[0] > nextSegment[0] and segment[1] > nextSegment[1]):
                    self.ax.plot([segment[2], segment[0]+.1], [segment[3], segment[3]], color='black')
                    self.ax.plot([segment[0]+.1, segment[0]+.1], [segment[3], nextSegment[3]], color='black')
                    self.ax.plot([segment[0]+.1, nextSegment[2]], [nextSegment[3], nextSegment[3]], color='black')
                elif segment[1] < nextSegment[1] or segment[0] > nextSegment[0]:
                    self.ax.plot([segment[2], segment[2]], [segment[3], nextSegment[3]], color='black')
                    self.ax.plot([segment[2], nextSegment[2]], [nextSegment[3], nextSegment[3]], color='black')
                else:
                    self.ax.plot([segment[2], nextSegment[2]], [segment[3], segment[3]], color='black')
                    self.ax.plot([nextSegment[2], nextSegment[2]], [segment[3], nextSegment[3]], color='black')

  def getBounds(self, coor):
    curr =  [ob for ob in self.obstacles if coor[1] >= (ob[3] - ob[1])  and coor[1] <= (ob[3] + ob[1])]
    left = self.startx
    right = self.endx
    for ob in curr:
        leftSide = ob[2]-ob[0]-.1
        rightSide = ob[2]+ob[0]+.1
        if rightSide < coor[0] :
            left = rightSide
        elif leftSide > coor[0]:
            right = leftSide
            break
    coor.append(left)
    coor.append(right)

  def cellDecomposition(self, sweepLines):
    ret = []
    step = int((self.endy-self.starty)/sweepLines)
    s_ins = g_ins = False
    self.getBounds(self.start)
    self.getBounds(self.goal)
    for i in range(self.starty, self.endy+1, step):
        curr =  [ob for ob in self.obstacles if i >= (ob[3] - ob[1])  and i <= (ob[3] + ob[1])]
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
        for l in ls:
            self.ax.plot(l[:2], [i,i], color='green', marker='o', linewidth=.25, markersize=1)
            self.ax.plot((l[1]+l[0])/2, i, color='black', marker='o', markersize=2)
        ret.append(ls)
        if not s_ins and self.start[1] < i+step:
            ret.append([[self.start[2],self.start[3], self.start[0], self.start[1], True]])
            s_ins=True
        if not g_ins and self.goal[1] < i+step:
            ret.append([[self.goal[2],self.goal[3], self.goal[0], self.goal[1], True]])
            g_ins=True
    return ret

  def printMap(self):
    for ob in self.obstacles:
        self.printSuperEllipse(self.ax, ob[0], ob[1], 20, ob[2], ob[3])
    plt.show()

# importing the required libraries 
import matplotlib.pyplot as plt 
from math import sin, cos, pi 
from matplotlib.patches import Ellipse
import math
import numpy
def rot(sigma):
    return numpy.asarray([[math.cos(sigma), -math.sin(sigma)],[math.sin(sigma), math.cos(sigma)]])
def sgn(x): 
    return ((x>0)-(x<0))*1 
def contains(ob, coor):
    lside = ob[2]-ob[0]
    rside = ob[2]+ob[0]
    top = ob[3]+ob[1]
    bottom = ob[3]-ob[1]
    return coor[0] >= bottom and coor[0] <= top and coor[1] <= rside and coor[1] >= lside
start = [-13,-8]
goal = [-36,10]
ma = Map(plt=plt)
# for i in range(50):
#     obstacle = (random.randint(2,5),random.randint(2,5),random.randint(-40,40),random.randint(-40,40))
#     if contains(obstacle, start) or contains(obstacle, goal):
#         i-=1
#         continue
#     ma.addObstacle(obstacle)


######## FLY CATCHER OBSTACLES ###############
ma.addObstacle((20,3,0,-16))
ma.addObstacle((20,3,0,6))
ma.addObstacle((3,16, -18,-5))

ma.addObstacle((3,9, 18,-17))
ma.addObstacle((3,9, 18,7))

ma.addObstacle((8,1, 18,-3))
ma.addObstacle((8,1, 18,-9))

ma.addStart(start)
ma.addGoal(goal)
# for i in range(11):
#     e = Ellipse(xy=[-i*3,-i*3], width=2, height=4, angle=18*i)
#     ma.addRobot(e)
decomp = ma.cellDecomposition(40)
ma.connectVerticesWithinCLayer(decomp)
ma.printMap()

# printSuperEllipse(ax, 5,5,4, 10, 10)
# 