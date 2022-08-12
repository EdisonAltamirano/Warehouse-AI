import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch 
from GPyOpt.methods import BayesianOptimization
from bayesOpt import *
import datetime
import distutils.util
from pyModbusTCP.client import ModbusClient
import time
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase_admin import firestore
import logging
import threading
import time
import requests
# cred = credentials.Certificate("google_services.json")
# firebase_admin.initialize_app(cred)
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
#################################
#   Define parameters manually #
#################################
def mapping(x,game,ros):
    return (x*game)/ros

class Game:
    """ Initialize PyGAME """    
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('WarehouseAI')
        self.Icon = pygame.image.load('img/LogoSF.jpg')
        pygame.display.set_icon(self.Icon)
        self.game_width = game_width
        self.game_height = game_height
        #Mission array -> [0]=pygame, [1]=RobotCoordinate
        #11->Negative
        #10->Positive
        self.missions = [[[584,176],[11412,10106]],[[355,175],[10212,10078]],[[528,405],[11206,10717]]]
        self.charging_station = [[[681,302],[11591,10534]]]
        self.modula_coord = [[[160,377],[11591,10534]]]
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background1.png")
        self.crash = False
        self.robot = Robot(self)
        self.food = Food(self)
        self.score = 0
        self.games = 0
        self.steps = 0

def display_ui(game, score, record,player1):
    myfont = pygame.font.SysFont('Warehouse AI', 20)
    myfont_bold = pygame.font.SysFont('Warehouse AI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    battery_text = myfont.render('P1 Battery: ', True, (0, 0, 0))
    battery_number = myfont_bold.render(str(player1.battery), True, (0, 0, 0))
    text_status = myfont.render('Status: ', True, (0, 0, 0))
    text_status_number = myfont_bold.render(str(player1.status), True, (0, 0, 0))
    game_status = myfont.render('Game: ', True, (0, 0, 0))
    game_status_number = myfont_bold.render(str(game.games), True, (0, 0, 0))
    steps_status = myfont.render('Steps: ', True, (0, 0, 0))
    steps_status_number = myfont_bold.render(str(game.steps), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 540))
    game.gameDisplay.blit(text_score_number, (120, 540))
    game.gameDisplay.blit(text_highest, (190, 540))
    game.gameDisplay.blit(text_highest_number, (350, 540))
    game.gameDisplay.blit(battery_text, (385, 540))
    game.gameDisplay.blit(battery_number, (470, 540))
    game.gameDisplay.blit(text_status, (540, 540))
    game.gameDisplay.blit(text_status_number, (615, 540))
    game.gameDisplay.blit(game_status, (45, 570))
    game.gameDisplay.blit(game_status_number, (120, 570))
    game.gameDisplay.blit(steps_status, (190, 570))
    game.gameDisplay.blit(steps_status_number, (350, 570))
    game.gameDisplay.blit(game.bg, (10, 10))
def update_screen():
    pygame.display.update()
def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record,player)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


class Robot(object):
    def __init__(self, game):
        x = random.uniform(0.1,0.9) * game.game_width
        y = random.uniform(0.1,0.9) * game.game_height - 30
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.image = pygame.image.load('img/snakeBody.png')
        self.food = 1
        self.eaten = False
        self.x_change = 20
        self.y_change = 0
        self.robot_status="Free"
        self.status=3
        #1:MissionStatus 0-Charge, 1-Move, 2-Free
        #2: #Goalx
        #3: #Goaly
        #4: Battery
        #5: RobotStatus 0-Charging, 1:Moving, 2:Free,3-Success,4-Failure
        #6: Distance
        #7: Locationx robot
        #8: Locationy robot
        #9: Goalxpi robot
        #10: Goalypi robot
        #11: Locationxpi robot
        #12: Locationypi robot
        self.inforobot=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.prevx_robot=-1
        self.pick_material=-1
        self.battery=100
        self.objectives=[]
        for i in game.missions:
            self.objectives.append([game.missions[1][0],game.missions[1][1]])
        self.charge=game.charging_station[0][1]
        self.modula_coord=game.modula_coord[0][1]
        #self.changeobjective=0

    def update_position(self, x, y):
        # if self.position[-1][0] != x or self.position[-1][1] != y:
        #     if self.food > 1:
        #         for i in range(0, self.food - 1):
        #             self.position[i][0], self.position[i][1] = self.position[i + 1]
        self.position[-1][0] = x
        self.position[-1][1] = y
    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y
        if game.crash == False:
            # for i in range(food):
            #     x_temp, y_temp = self.position[len(self.position) - 1 - i]
            game.gameDisplay.blit(self.image, (self.position[0][0], self.position[0][1]))
            update_screen()
        else:
            pygame.time.wait(300)

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]
        #Cloud update battery
        # doc_ref = self.db.collection(u'Robots').document(u'DashgoB1')
        if self.eaten:
            #self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
            #print("Comiiii")
        #Modula Priority , change coordinate 
        try:
            modula_resp = requests.get('http://10.22.229.191/Modula/api/Picking')
            dict_resp=modula_resp.json()
        except:
            dict_resp={}

        if np.array_equal(move, [0, 0, 1]):
            # print("Free robot")
            self.inforobot[0]=2
            # move_array = self.x_change, self.y_change
            # self.x_change, self.y_change = move_array
            # self.x = x + self.x_change
            # self.y = y + self.y_change
            self.x = x 
            self.y = y 
            #Free robot
            #self.battery-=2.5
            self.image=pygame.image.load('img/blueBody.png')
            self.status=3
            # doc_ref.update({
            # u'mission': "Nothing"
            # })

        elif np.array_equal(move, [0, 1, 0]):  # right - going horizontal
            #Moving robot
            self.inforobot[0]=1
            #Goalx,y
            
            if self.pick_material==1:
                self.pick_material=2
                requests.post('http://10.22.229.191/Modula/api/ConfirmarPicking')

            if len(dict_resp)>0 :
                #Update XArm coordinates movement
                self.inforobot[1]=self.modula_coord[0]
                self.inforobot[2]=self.modula_coord[1]
                self.pick_material=1
            else:
                self.inforobot[1]=food.x_robotfood
                self.inforobot[2]=food.y_robotfood

            self.image=pygame.image.load('img/snakeBody.png')
            #Assume time navigationSS
            #Reduce Battery
            #Update location to objective if success 
            if(self.inforobot[4]==3):
                self.x = food.x_food -20    
                self.y = food.y_food -20 


            self.status=1
            # doc_ref.update({
            # u'mission': "Move"
            # })
            #move_array = [0, self.x_change]
        elif np.array_equal(move, [1, 0, 0]) :  # right - going vertical
            #Charging robot
            self.inforobot[0]=0
            self.inforobot[1]=self.charge[0]
            self.inforobot[2]=self.charge[1]
            self.image=pygame.image.load('img/redBody.png')
            #Add energy
            self.status=2
            # doc_ref.update({
            # u'mission': "Charge"
            # })
            #move_array = [-self.y_change, 0]

        # self.x_change, self.y_change = move_array
        # self.x = x + self.x_change
        # self.y = y + self.y_change

        self.update_position(self.x, self.y)


class Food(object):
    def __init__(self,game):
        #self.x_food = random.uniform(0.1,0.9)* game.game_width
        #self.y_food = random.uniform(0.1,0.9)* game.game_height
        self.pyrandom = random.choice(game.missions)
        self.prevx_robotfood=-1
        #self.prevy_coord=-1
        self.x_food = self.pyrandom[0][0]
        self.y_food = self.pyrandom[0][1]
        self.x_robotfood=self.pyrandom[1][0]
        self.y_robotfood=self.pyrandom[1][1]
        #Goal Robot x,y
        game.robot.inforobot[1] = self.x_robotfood
        game.robot.inforobot[2] = self.y_robotfood
        #PI /Locationx,y
        game.robot.inforobot[8] = self.x_food
        game.robot.inforobot[9] = game.game_height-self.y_food

        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        pyrandom = random.choice(game.missions)
        self.x_food = pyrandom[0][0]
        self.y_food = pyrandom[0][1]
        #Goal Robot x,y
        game.robot.inforobot[1] = pyrandom[1][0]
        game.robot.inforobot[2] = pyrandom[1][1]
        #PI /Locationx,y
        game.robot.inforobot[8] = self.x_food
        game.robot.inforobot[9] = game.game_height-self.y_food
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)   
    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen() 

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=100)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    SERVER_HOST = "10.22.240.51"
    SERVER_PORT = 12345
    c = ModbusClient()
    c.host(SERVER_HOST)
    c.port(SERVER_PORT)
    game = Game(760, 570)
    record = 0
    while True:
        if not c.is_open():
                    if not c.open():
                        print("unable to connect to "+SERVER_HOST+":"+str(SERVER_PORT))
        if c.is_open():
            '''
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:  # or MOUSEBUTTONDOWN depending on what you want.
                    print(event.pos)
            '''
            food1 = game.food
            display(game.robot, food1, game, record)
            #Read Battery / RobotStatus /Distance (Modbus)
            bits = c.read_holding_registers(0, 12)
            game.robot.inforobot[3]=bits[3]
            game.robot.inforobot[4] = bits[4]
            game.robot.inforobot[5] = bits[5]
            #Map coordinate location to pi
            theta = np.radians(13)
            offsetx=7.9115512
            offsety=2.50132282
            rheight=11.29203556
            rwidth=17.69995916
            c1, s = np.cos(theta), np.sin(theta)
            R = np.array(((c1, -s), (s, c1)))
            #Convert bit +-
            game.robot.inforobot[10]=int(mapping(bits[6]+offsetx,game.game_width,rwidth))
            game.robot.inforobot[11]=int(mapping(bits[7]+offsety,game.game_height,rheight))
            time.sleep(1)
            #print(bits)
            #Update registers MissionStatus,Goalx,Goaly
            c.write_multiple_registers(0,[bits[0],game.robot.inforobot[1],game.robot.inforobot[2]])
            c.write_multiple_registers(8,[game.robot.inforobot[8],game.robot.inforobot[9],game.robot.inforobot[10],game.robot.inforobot[11]])
            
            time.sleep(1)