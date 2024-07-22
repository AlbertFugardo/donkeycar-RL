import math
import numpy as np
import time

class Reward():
    def __init__(self, center_line):
        self.lap_count = 0
        self.center_line = center_line
        
        _, cum_dist = self.distance([x[0] for x in center_line],[x[1] for x in center_line]) # x and y points
        self.track_progress = [100*x/cum_dist[-1] for x in cum_dist]
        if self.track_progress[-1] - 100.0 > 0.001:
            print(f"Something has gone wrong in the Reward init! The last track progress should be 100 but is {self.track_progress[-1]}")
            
        self.last_step_progress = 0
        
        # variables for sector time reward
        self.sectors = {}
        self.sectors["idx"] = [100,180,250,340,430,520,600,670,720,780,840,900,960,1020,1080,1180,1280,1360]
        self.sectors["is_up"] = []
        self.sectors["m"] = []
        self.sectors["c"] = []
        
        a1 = [pt[0] for pt in center_line]
        b1 = [pt[1] for pt in center_line]
        for index in self.sectors["idx"]:
            point = (a1[index]+np.random.normal(0, 0.5, 1)[0],b1[index]+np.random.normal(0, 0.5, 1)[0])
            _,m,c,is_up = self.is_ahead(index, point, track=(a1,b1))
            self.sectors["is_up"].append(is_up)
            self.sectors["m"].append(m)
            self.sectors["c"].append(c)
    
        self.next_sector_idx = 0
        self.time_from_prev_sector = time.time()
        
    @staticmethod
    def distance(x,y): # given a lists of x and y points, returns its distances and cumulative distances
        n = len(x) # == len(y)
        dist = [0] # distance
        cum_dist = [0] # cumulative distance
        for i in range(1,n):
            dist.append( np.sqrt ( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) )
            cum_dist.append( cum_dist[i-1]+dist[i] )
        return dist, cum_dist

    
    def reward_test(self, self_sim, done: bool) -> float:
        # a new lap is completed
        if self_sim.lap_count > self.lap_count:
            self.lap_count = self_sim.lap_count
            time = self_sim.last_lap_time
            return (100/time)**2
        # episode over, the lap has not been completed
        if done:
            self.lap_count = 0
            return -100.0
        
        # Collision
        if self_sim.hit != "none":
            return -2.0

        # if forward vel give some reward
        if self_sim.forward_vel > 0.1:
            return 0.1

        # slow forward velocity or reverse
        return -0.1
    
    def reward_test2(self, self_sim, done: bool) -> float:
        # a new lap is completed
        if self_sim.lap_count > self.lap_count:
            self.lap_count = self_sim.lap_count
            time = self_sim.last_lap_time
            return (100/time)**2
        # episode over, the lap has not been completed
        if done:
            self.lap_count = 0
            return -10 - self_sim.forward_vel
        
        # Collision
        if self_sim.hit != "none":
            return -20.0

        # if forward vel give some reward
        return 1 + self_sim.forward_vel
    
    def reward_test3(self, self_sim, done: bool) -> float:
        # a new lap is completed
        if self_sim.lap_count > self.lap_count:
            self.lap_count = self_sim.lap_count
            time = self_sim.last_lap_time
            return 8*(20-time)+50

        # episode over, the lap has not been completed
        if done:
            self.lap_count = 0
            return -10 - self_sim.forward_vel
        
        # Collision
        if self_sim.hit != "none":
            return -20.0

        # if forward vel give some reward
        return 1 + self_sim.forward_vel
    
    
    @staticmethod
    def progress(point,center_line,track_progress):
        # index of the center line point closest to "point"
        idx = min(range(len(center_line)), key=lambda i: abs( np.sqrt((center_line[i][0]-point[0])**2 + (center_line[i][1]-point[1])**2) ))
        return track_progress[idx]

    def progress_reward(self, self_sim, done: bool) -> float:
        car_x = self_sim.x
        car_y = self_sim.z
        step_progress = self.progress((car_x,car_y),self.center_line,self.track_progress)
        
        if done or self_sim.hit != "none": # lap not completed
            self.last_step_progress = 0
            return -10 - self_sim.forward_vel
        rew = step_progress - self.last_step_progress
        self.last_step_progress = step_progress
        return rew
    
    def progress_reward2(self, self_sim, done: bool) -> float:
        car_x = self_sim.x
        car_y = self_sim.z
        step_progress = self.progress((car_x,car_y),self.center_line,self.track_progress)
        
        if done or self_sim.hit != "none": # lap not completed
            self.last_step_progress = 0
            return -1
        rew = step_progress - self.last_step_progress
        self.last_step_progress = step_progress
        return rew * self_sim.forward_vel
    
    # auxiliar functions for time_sector_reward
    @staticmethod
    def create_perp_line(track, index):
        """
        Given a line track index, it returns the parameters m and c
        of its perpendicular straight line
        """
        a1 = track[0]
        b1 = track[1]
        x0 = a1[index-1]; x1 = a1[index]; x2 = a1[(index+1) % len(a1)]
        y0 = b1[index-1]; y1 = b1[index]; y2 = b1[(index+1) % len(b1)]
        m = (y2-y0)/(x2-x0)

        m_perp = -1/m
        c_perp = y2-m_perp*x2
        return m_perp, c_perp

    def is_ahead(self, line_point_idx, point, track):
        """
        Given a line point index, any point and the track, returns if the 
        point is ahead in the track in comparison to the line point.
        Note: it only works if the point is very close to the line_point_idx
        """
        x_track = track[0]; y_track = track[1]
        m,c = self.create_perp_line(track,line_point_idx) # creates a perpendicular line from the line point
        
        # we get the direction in which the track is pointing (if the next line point is above the perpendicular line or below)
        is_up = False
        if y_track[(line_point_idx+4) % len(x_track)] > (m*x_track[(line_point_idx+4) % len(x_track)]+c):
            is_up = True
        # point_idx[1] is its y, and point_idx[0] its x
        if is_up and point[1] > (m*point[0]+c):
            return True,m,c,is_up
        if not is_up and point[1] < (m*point[0]+c):
            return True,m,c,is_up
        return False,m,c,is_up
    
    @staticmethod
    def is_ahead_simplified(sector_idx, point, sectors):
        """
        Given a sector index, any point and the sectors information, it returns if the 
        point is ahead in the track in comparison to the sector point.
        Note: it only works if the point is close to the sector
        """
        m = sectors["m"][sector_idx]
        c = sectors["c"][sector_idx]
        is_up = sectors["is_up"][sector_idx]
        
        # point_idx[1] is its y, and point_idx[0] its x
        if is_up and point[1] > (m*point[0]+c):
            return True
        if not is_up and point[1] < (m*point[0]+c):
            return True
        return False
    
    def time_sector_reward(self, self_sim, done: bool) -> float:
        if done or self_sim.hit != "none": # lap not completed
            self.next_sector_idx = 0
            self.time_from_prev_sector = time.time()
            return -1
        
        car_x = self_sim.x
        car_y = self_sim.z
        rew = 0
        if self.is_ahead_simplified(self.next_sector_idx, (car_x,car_y), self.sectors): # if the car is ahead of the coming sector
            rew = 1/(time.time()-self.time_from_prev_sector)
            self.time_from_prev_sector = time.time()
            self.next_sector_idx = (self.next_sector_idx + 1) % len(self.sectors["idx"])
        return self_sim.forward_vel/10 + rew