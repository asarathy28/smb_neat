import retro


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')


done = True
for step in range(500):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    print(action)
    state, reward, done, info = env.step(action)
    env.render()

env.close()


#don't have the ROM...
