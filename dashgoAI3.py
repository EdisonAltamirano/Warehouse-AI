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
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    #params['episodes'] = 100    
    params['episodes'] = 100
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = False #True
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


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
        self.player = Player(self)
        self.player2=Player(self)
        self.food = Food(self)
        self.score = 0
        self.games = 0
        self.steps = 0


class Player(object):
    def __init__(self, game):
        x = random.uniform(0.1,0.9) * game.game_width
        y = random.uniform(0.1,0.9) * game.game_height - 30
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0
        self.robot_status="Free"
        self.status=3
        # cred = credentials.Certificate("google_services.json")
        # firebase_admin.initialize_app(cred)
        # self.db = firestore.client()
        #1:MissionStatus 0-Charge, 1-Move, 2-Free
        #2: #Goalx
        #3: #Goaly
        #4: Battery
        #5: RobotStatus 0-Charging, 1:Moving, 2:Free,3-Success,4-Failure
        #6: Distance
        self.inforobot=[0,0,0,0,0,0]
        self.pick_material=-1
        self.battery=100
        self.objectives=[]
        for i in game.missions:
            self.objectives.append([game.missions[1][0],game.missions[1][1]])
        self.charge=game.charging_station[0][1]
        self.modula_coord=game.modula_coord[0][1]
        self.changeobjective=0

    def update_position(self, x, y):
        # if self.position[-1][0] != x or self.position[-1][1] != y:
        #     if self.food > 1:
        #         for i in range(0, self.food - 1):
        #             self.position[i][0], self.position[i][1] = self.position[i + 1]
        self.position[-1][0] = x
        self.position[-1][1] = y

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
        modula_resp = requests.get('http://10.22.229.191/Modula/api/Picking')
        try:
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
            if self.changeobjective<=1:
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
                if self.changeobjective==1:
                    self.changeobjective=0


            self.image=pygame.image.load('img/snakeBody.png')
            #Assume time navigationSS
            #Reduce Battery
            #Update location to objective if success 
            if(self.inforobot[5]==3):
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

        if ((self.battery<30)):
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

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


class Food(object):
    def __init__(self,game):
        #self.x_food = random.uniform(0.1,0.9)* game.game_width
        #self.y_food = random.uniform(0.1,0.9)* game.game_height
        self.pyrandom = random.choice(game.missions)
        self.x_food = self.pyrandom[0][0]
        self.y_food = self.pyrandom[0][1]
        self.x_robotfood=self.pyrandom[1][0]
        self.y_robotfood=self.pyrandom[1][1]
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food -20 and player.y == food.y_food -20:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


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



def display(player,player2, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record,player)
    #player2.display_player(player2.position[-1][0], player2.position[-1][1], player.food, game)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, player2,game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

def worker(arg):
    while not arg["stop"]:
        logging.debug("worker thread checking in")
        # db = firestore.client()
        # doc_ref = db.collection(u'Robots').document(u'DashgoB1')
        # snapshot = doc_ref.get()
        # inforobot = snapshot.to_dict()
        # arg["robot1"].battery=inforobot['battery']
        # arg["robot1"].robot_status=inforobot['robotstatus']
        time.sleep(3)
def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    info = {"stop": False,"robot1":"player1"}
    SERVER_HOST = "10.22.240.51"
    SERVER_PORT = 12345
    c = ModbusClient()
    c.host(SERVER_HOST)
    c.port(SERVER_PORT)
    # thread = threading.Thread(target=worker, args=(info,))
    while counter_games < params['episodes']and total_score<=200:
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        '''        
        # Initialize classes
        #game = Game(760, 540)
        game = Game(760, 570)
        player1 = game.player
        player2 = game.player2
        food1 = game.food

        # Perform first move
        initialize_game(player1,player2, game, food1, agent, params['batch_size'])
        if params['display']:
            display(player1,player2, food1, game, record)
        counter_games+=1
        game.games += counter_games
        #pygame.time.wait(1000)
        steps = 0       # steps since the last positive reward
        player1.battery=100
        while ((not game.crash) and (steps < 240))and game.score<=200:
            if not c.is_open():
                if not c.open():
                    print("unable to connect to "+SERVER_HOST+":"+str(SERVER_PORT))
            if c.is_open():
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONUP:  # or MOUSEBUTTONDOWN depending on what you want.
                        print(event.pos)
                #Read Battery / RobotStatus /Distance (Modbus)
                bits = c.read_holding_registers(0, 6)
                player1.inforobot[3]=bits[3]
                player1.inforobot[4] = bits[4]
                player1.inforobot[5] = bits[5]
                time.sleep(1)
                #print(bits)
                #Update registers MissionStatus,Goalx,Goaly
                c.write_multiple_registers(0,[player1.inforobot[0],player1.inforobot[1],player1.inforobot[2]])
                time.sleep(1)
            game.steps = steps
            # perform new move and get new state
            if player1.inforobot[5] == 3 or player1.inforobot[5]==4 or steps==0:#Failure or success then other move and set step to 0
                if not params['train']:
                    agent.epsilon = 0.1
                else:
                    # agent.epsilon is set to give randomness to actions
                    agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

                # get old state
                state_old = agent.get_state(game, player1, food1)

                # perform random actions based on agent.epsilon, or choose the action
                if random.uniform(0, 1) < agent.epsilon:
                    final_move = np.eye(3)[randint(0,2)]
                else:
                    # predict action based on the old state
                    with torch.no_grad():
                        state_old_tensor = torch.tensor(state_old.reshape((1, 4)), dtype=torch.float32).to(DEVICE)
                        prediction = agent(state_old_tensor)
                        final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
                        # print("Prediction")
                        # print(final_move)
                player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
                state_new = agent.get_state(game, player1, food1)

                # set reward for the new state
                reward = agent.set_reward(player1, game.crash)
                
                # if food is eaten, steps is set to 0
                if reward > 0:
                    steps = 0
                    
                if params['train']:
                    # train short memory base on the new action and state
                    agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                    # store the new data into a long term memory
                    agent.remember(state_old, final_move, reward, state_new, game.crash)

                record = get_record(game.score, record)
            if params['display']:
                display(player1, player2,food1, game, record)
                pygame.time.wait(params['speed'])
                # if player1.eaten:
                #     print("Comioooooooooooo")
                #     pygame.time.wait(1000)
            steps+=1
      
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    if params['plot_score']:
        print("I am plottinggg")
        plot_seaborn(counter_plot, score_plot, params['train'])
    if params['train']:
        model_weights = agent.state_dict()
        print("Saving modellllllllll")
        info["stop"] = True
        #thread.join()
        torch.save(model_weights, params["weights_path"])
        return total_score
    else:
        mean, stdev = get_mean_stdev(score_plot)
        info["stop"] = True
        #thread.join()
        return total_score, mean, stdev
    

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=100)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed
    params['display'] = args.display
    params['speed'] = args.speed
    SERVER_HOST = "10.22.240.51"
    SERVER_PORT = 12345
    c = ModbusClient()
    c.host(SERVER_HOST)
    c.port(SERVER_PORT)
    while True:
        if not c.is_open():
                    if not c.open():
                        print("unable to connect to "+SERVER_HOST+":"+str(SERVER_PORT))
        if c.is_open():
            if args.bayesianopt:
                bayesOpt = BayesianOptimizer(params)
                bayesOpt.optimize_RL()
            if params['train']:
                print("Training...")
                params['load_weights'] = False   # when training, the network is not pre-trained
                run(params)
            if params['test']:
                print("Testing...")
                params['train'] = False
                params['load_weights'] = True
                run(params)