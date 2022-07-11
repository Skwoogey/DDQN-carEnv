# import the pygame module, so you can use it
import copy
import pygame
import numpy as np

class Vector2:
    def __init__(self, x=None, y=None):
        #print(x, y, type(x))
        if x == None and y == None:
            #print("empty vector")
            self.x = 0.0
            self.y = 0.0
        elif (type(x) == float or type(x) == np.float64)  and y == None:
            #print(type(x), x)
            self.x = np.cos(x)
            self.y = np.sin(x)
            #print(self.x, self.y)
        else:
            #print("initialized vector")
            self.x = x
            self.y = y

    def __add__(self, a):
        cp = Vector2(self.x, self.y)
        cp.x += a.x
        cp.y += a.y
        return cp

    def __sub__(self, a):
        cp = Vector2(self.x, self.y)
        cp.x -= a.x
        cp.y -= a.y
        return cp

    def scale(self, scl):
        newVec = Vector2(self.x, self.y)
        newVec.x *= scl
        newVec.y *= scl
        return newVec

    def scaleVec(self, sclVec):
        newVec = Vector2(self.x, self.y)
        newVec.x *= scl.x
        newVec.y *= scl.y
        return newVec

    def getTuple(self):
        return (self.x, self.y)

    def magnitude(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def print(self):
        print('(' + str(self.x) + ', ' + str(self.y) + ')')


class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

    def __mul__(s, o):
        x13 = s.pt1[0] - o.pt1[0]
        x12 = s.pt1[0] - s.pt2[0]
        x34 = o.pt1[0] - o.pt2[0]
        y34 = o.pt1[1] - o.pt2[1]
        y13 = s.pt1[1] - o.pt1[1]
        y12 = s.pt1[1] - s.pt2[1]

        denom = (x12 * y34)- (y12 * x34)

        if denom < 0.001 and denom > -0.001:
            #print("denom")
            return None

        t = ((x13 * y34) - (y13 * x34)) / denom
        if t < 0.0 or t > 1.0:
            #print("t")
            return None

        u = -((x12 * y13)- (y12 * x13)) / denom
        if u < 0.0 or u > 1.0:
            #print("u")
            return None
        
        return (s.pt1[0] - t * x12, s.pt1[1] - t * y12)

    def draw(self, screen):
        pygame.draw.line(screen, (0, 0, 255), self.pt1, self.pt2, 1)


class Lines:
    def __init__(self):
        self.pts = []
        self.lines = []

    def addPoint(self, pt):
        self.pts.append(pt)

    def addLine(self, line):
        self.lines.append(line)

    def fromFile(file):
        lines = []
        try: 
            track_file = open(file, "r")
        except:
            print("No such file: " + file)
            exit()

        track_lines = track_file.readlines()
        track_file.close()
        if track_lines != None:
            for line in track_lines:
                #print(line)
                if line == "\n":
                    continue
                    #print("lineskip")
                elif line =="[\n":
                    #print("linestart")
                    lines.append(Lines())
                elif line == "]\n":
                    lines[-1].calculateLines()
                else:
                    #print("lineadd")
                    pt = [float(num) for num in line.split(" ")]
                    pt = (pt[0], pt[1])
                    #print(pt)
                    lines[-1].addPoint(pt)
        return lines

    def draw(self, screen):
        if len(self.pts) > 1:
            pygame.draw.lines(screen, 0, False, self.pts, 2)
        else:
            for line in self.lines:
                line.draw(screen) 

    def calculateLines(self):
        self.lines = []
        for pti in range(len(self.pts) - 1):
            self.lines.append(Line(self.pts[pti], self.pts[pti + 1]))

    def intersect(self, other):
        for oline in other.lines:
            for sline in self.lines:
                pt = oline * sline
                if pt != None:
                    return True
        return False

    def collidePoint(self, other):
        contacts = []
        for oline in other.lines:
            for sline in self.lines:
                pt = oline * sline
                if pt != None:
                    contacts.append(pt)

        return contacts

    def castLine(self, other):
        contact = Vector2(other.pt2[0] - other.pt1[0], 
                            other.pt2[1] - other.pt1[1]).magnitude()
        cpt = None
        for sline in self.lines:
            pt = other * sline
            if pt != None:
                nContact = Vector2(pt[0] - other.pt1[0], 
                                    pt[1] - other.pt1[1]).magnitude()
                if  nContact < contact:
                    contact = nContact
                    cpt = pt

        return contact, cpt


class Car:
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size.scale(0.5)
        self.velocity = Vector2()
        self.angle = 0.0
        self.maxSpeed = 5.0
        self.color = pygame.Color(50, 50, 50)
        self.sensors = []

    def getInput(self, input):
        self.angle += input.x * 0.02 * self.velocity.magnitude()
        dirVec = Vector2(self.angle)
        self.velocity += dirVec.scale(input.y * 0.2)
        self.pos += self.velocity
        self.velocity = self.velocity.scale(0.95)

        mag = self.velocity.magnitude()
        if mag > self.maxSpeed:
            self.velocity = self.velocity.scale(self.maxSpeed / mag)

    def calculateLines(self):
        self.fwdVec = Vector2(self.angle)
        self.rightVec = Vector2(self.fwdVec.y, -self.fwdVec.x)
        #fwdVec.print()
        carFwdVec = self.fwdVec.scale(self.size.y)
        carRightVec = self.rightVec.scale(self.size.x)
        self.pts = []
        self.pts.append(self.pos + carFwdVec + carRightVec)
        self.pts.append(self.pos + carFwdVec - carRightVec)
        self.pts.append(self.pos - carFwdVec - carRightVec)
        self.pts.append(self.pos - carFwdVec + carRightVec)

    def getLines(self):
        lines = Lines()
        for pt in self.pts:
            lines.addPoint(pt.getTuple())
        lines.addPoint(self.pts[0].getTuple())
        lines.calculateLines()
        return lines

    def attachSensor(self, sensor):
        self.sensors.append(sensor)

    def getSensors(self):
        sensors = Lines()
        for sensor in self.sensors:
            sPos = self.pos + (self.fwdVec.scale(sensor.pos[1])) + (self.rightVec.scale(sensor.pos[0]))
            sDir = Vector2(self.angle + sensor.angle)
            sensors.addLine(Line(sPos.getTuple(), (sPos + sDir.scale(sensor.distance)).getTuple()))

        return sensors

    def draw(self, screen):
        pygame.draw.polygon(screen, self.color, [pt.getTuple() for pt in self.pts])

class Sensor:
    def __init__(self, pos, angle, distance):
        self.pos = pos
        self.angle = angle
        self.distance = distance

class AS:
    def __init__(self, num):
        self.num = num

    def sample(self):
        return np.random.choice(self.num)

class CarEnvironment:
    def __init__(self, track, rewardGates, carPos, carAngle):
        self.renderWindow = True
        self.renderCar = True
        self.renderTrack = True
        self.renderGates = True
        self.renderSensors = True
        self.renderContacts = True

        self.action = AS(7)

        self.humanInteraction = False

        self.carStartState = (carPos, carAngle)
        self.track = Lines.fromFile(track)
        self.rewardGates = Lines.fromFile(rewardGates)
        self.gateCount = len(self.rewardGates)
        self.car = Car(Vector2(), Vector2(12, 20))
        #straight
        self.car.attachSensor(Sensor((0, 8), 0, 200))
        #15 degrees forward
        self.car.attachSensor(Sensor((-4, 8), 0.261799, 200))
        self.car.attachSensor(Sensor((4, 8), -0.261799, 200))
        #45 degrees forward
        self.car.attachSensor(Sensor((-4, 8), 0.785398, 200))
        self.car.attachSensor(Sensor((4, 8), -0.785398, 200))
        #90 degrees forward
        self.car.attachSensor(Sensor((-4, 8), 1.5708, 200))
        self.car.attachSensor(Sensor((4, 8), -1.5708, 200))
        #45 degrees backward
        self.car.attachSensor(Sensor((-4, -8), 2.35619, 200))
        self.car.attachSensor(Sensor((4, -8), -2.35619, 200))
        #backward
        self.car.attachSensor(Sensor((0, -8), 3.14159, 200))
        self.action = Vector2()
        self.reset()
        self.init_screen()

    def reset(self):
        self.car.pos, self.car.angle = self.carStartState
        self.car.velocity = Vector2()
        self.action = Vector2()
        self.rewarGateIndex = 0
        self.stepsSinceLastReward = 0
        self.LastDistToReward = self.distanceToRewardGate()
        self.updateState()
        return self.state


    def init_screen(self):
        pygame.init()
        # load and set the logo
        pygame.display.set_caption("minimal program")
         
        # create a surface on screen that has the size of 240 x 180
        self.screen_size = Vector2(1024, 800)
        self.screen = pygame.display.set_mode(self.screen_size.getTuple())
        self.WindowExists = True

    def step(self, action):
            if not self.humanInteraction:
                self.action = action

            for event in pygame.event.get():
                # only do something if the event is of type QUIT
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    self.WindowExists = False
                    pygame.quit()

                if self.humanInteraction:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.action.x = -1.0
                        elif event.key == pygame.K_RIGHT:
                            self.action.x = 1.0
                        elif event.key == pygame.K_UP:
                            self.action.y = 1.0
                        elif event.key == pygame.K_DOWN:
                            self.action.y = -1.0
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                            self.action.x = 0.0
                        elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                            self.action.y = 0.0

            self.updateState()

            #GetReward
            state = self.state
            reward = self.reward
            failed = self.failed
            return state, reward, failed, self.GateCleared
            
    def distanceToRewardGate(self):
        curRewardGate = self.rewardGates[self.rewarGateIndex]
        distance = Vector2(self.car.pos.x - curRewardGate.lines[0].pt1[0], 
                            self.car.pos.y - curRewardGate.lines[0].pt1[1]).magnitude()
                            
        distance += Vector2(self.car.pos.x - curRewardGate.lines[0].pt2[0], 
                            self.car.pos.y - curRewardGate.lines[0].pt2[1]).magnitude()
                            
        distance /= 2
        
        return distance

    def updateState(self, getGates=True):
        #Update Car and sensors
        self.car.getInput(self.action)
        self.car.calculateLines()
        self.carSensors = self.car.getSensors()
        self.carLines = self.car.getLines()

        #Get State
        self.failed = 0
        self.state = []
        self.contactPoints = []
        
        # relative speed
        rs = Vector2()
        rs.x = np.cos(-self.car.angle) * self.car.velocity.x - \
                np.sin(-self.car.angle) * self.car.velocity.y
                
        rs.y = np.sin(-self.car.angle) * self.car.velocity.x + \
                np.cos(-self.car.angle) * self.car.velocity.y
                
        self.state.append(rs.x)
        self.state.append(rs.y)
        
        for sLine in self.carSensors.lines:
            contact = 200.0
            cpt = None
            for trackLine in self.track:
                nContact, pt = trackLine.castLine(sLine)
                if nContact < contact:
                    contact = nContact
                    cpt = pt
            if cpt != None:
                self.contactPoints.append(cpt)
            #print("wall")
            self.state.append(contact / 200.0)

        for trackLine in self.track:
            if trackLine.intersect(self.carLines):
                self.reward = -1000
                self.failed = 1
                break

        self.GateCleared = False
        if self.rewardGates[self.rewarGateIndex].intersect(self.carLines) and not self.failed:
            self.rewarGateIndex += 1
            if self.rewarGateIndex == len(self.rewardGates):
                self.rewarGateIndex = 0
            self.reward = self.LastDistToReward
            self.stepsSinceLastReward = 0
            self.GateCleared = True
            self.LastDistToReward = self.distanceToRewardGate()

        if not self.GateCleared and not self.failed:
            newDistToReward = self.distanceToRewardGate()
            self.reward = self.LastDistToReward - newDistToReward
            self.LastDistToReward = newDistToReward
        
        for sLine in self.carSensors.lines:
            nContact, cpt = self.rewardGates[self.rewarGateIndex].castLine(sLine)
            if cpt != None:
                self.contactPoints.append(cpt)
            #print("gate")
            self.state.append(nContact / 200.0)

        if getGates == False:
            #print("cut")
            self.state = self.state[:10]

        self.stepsSinceLastReward += 1
       
        

    def render(self):
        if self.renderWindow and self.WindowExists:
            self.screen.fill((255, 255, 255))
            if self.renderTrack:
                for line in self.track:
                    line.draw(self.screen)
            if self.renderCar:
                self.car.draw(self.screen)
            if self.rewardGates:
                for line in self.rewardGates:
                    line.draw(self.screen)
            if self.renderSensors:
                for sLine in self.carSensors.lines:
                    sLine.draw(self.screen)
            if self.renderContacts:
                for cpt in self.contactPoints:
                    pygame.draw.circle(self.screen, (0, 0, 255), (int(cpt[0]), int(cpt[1])), 4, 4)
            pygame.display.update()

