# import the pygame module, so you can use it
import os, sys
import copy
import pygame
import numpy as np
import env

# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    # load and set the logo
    pygame.display.set_caption("minimal program")
     
    # create a surface on screen that has the size of 240 x 180
    screen_size = env.Vector2(1024, 800)
    screen = pygame.display.set_mode(screen_size.getTuple())
    clock = pygame.time.Clock()
     
    # define a variable to control the main loop
    running = True

    lines = env.Lines.fromFile(sys.argv[1])

    
    # main loop
    screen.fill((255, 255, 255))
    #print(lines)
    

    pygame.display.update()

    for line in lines:
        line.draw(screen)

    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False

        pygame.display.update()
        clock.tick(60)
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()