import numpy as np
import cv2
import neat
import pickle
import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)

def eval_genomes(genomes, config):

    for g_id, g in genomes:
        #observation variable
        ob = env.reset()
        #action varibale
        ac = env.action_space.sample()

        #net = neat.nn.recurrent.RecurrentNetwork.create(g, config)
        net = neat.nn.FeedForwardNetwork.create(g, config)

        #(width, height, color)
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        best_fitness = 0
        g.fitness = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        done = False
        while not done:

            imgarray = []

            env.render()
            frame += 1

            #shows what the AI sees
            """
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            scaledimg = cv2.resize(scaledimg, (inx, iny))
            cv2.imshow('main', scaledimg)
            cv2.waitKey(1)
            """

            #downsize the frame
            ob = cv2.resize(ob, (inx, iny))
            #greyscale the frame
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #reshape frame to fit neural network??
            ob = np.reshape(ob, (inx, iny))

            #coverts 2D array of pixels into 1D array of values
            for x in ob:
                for y in x:
                    imgarray.append(y)

            nn_output = net.activate(imgarray)
            #print ("nn_output is : " + str(nn_output))

            # using join() + list comprehension
            # converting binary list to integer
            #int_output = int("".join(str(round(x)) for x in nn_output), 2) + 1

            # using bit shift + | operator
            # converting binary list to integer
            int_output = 0
            for ele in nn_output:
                if ele > 0.5:
                    push = 1
                else:
                    push = 0

                int_output = (int_output << 1) | push

            int_output += 1
            #print(int_output)

            ob, reward, done, info = env.step(int_output)
            #print("current fitness: " + str(g.fitness))

            """
            xpos = info['x_pos']
            if xpos > xpos_max:
                 g.fitness += 1
                 xpos_max = xpos
            """

            #print("RAW type: " , type(reward)," value: ", str(reward))
            #print("itemed type: ", type(reward.item())," value: ", str(reward.item()))
            rew = 0
            if isinstance(reward, np.generic):
                rew = np.asscalar(reward)

            #print("RAW type: " , type(reward)," value: ", str(reward))
            #print("MOD type: " , type(rew)," value: ", str(rew))
            g.fitness += rew

            if g.fitness > best_fitness:
                best_fitness = g.fitness
                #counter -= 1
            else:
                counter += 1

            life = info['life']
            if life < 2:
                done = True

            if done or counter == 200:
                done = True

                print("GENOME " + str(g_id))
                print("FITNESS: " + str(g.fitness))
                g.fitness += info['score']
                print("FITNESS + SCORE: " + str(g.fitness))


def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes,100)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, out, 1)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "right-config-feedforward.txt")
    run(config_path)
