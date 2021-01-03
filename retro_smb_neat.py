import numpy as np
import cv2
import neat
import pickle
import os
import retro


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')



def eval_genomes(genomes, config):

    for g_id, g in genomes:

        #observation variable
        ob = env.reset()
        #action varibale
        ac = env.action_space.sample()

        #(width, height, color)
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(g, config)

        best_fittness = 0
        current_fittness = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        while not done:

            imgarray = []

            env.render()
            frame += 1

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
            print(nn_output)

            ob, reward, done, info = env.step(nn_output)
            print(info)

            xpos = info['xscrollLo']
            if xpos > xpos_max:
                 current_fittness += 1
                 xpos = xpos_max

            if current_fittness > best_fittness:
                best_fittness = current_fittness
            else:
                counter += 1

            if done or counter == 200:
                done = True
                print(g_id, current_fittness)

            g.fittness = current_fittness



def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "retroSMB-config-feedforward.txt")
    run(config_path)
