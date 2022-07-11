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

    refLines = []
    if len(sys.argv) > 1:
    	for arg in sys.argv[1:]:
    		refLines = refLines + env.Lines.fromFile(arg)
     
    # define a variable to control the main loop
    running = True

    lines = []
    ongoingLine = False
     
    # main loop
    screen.fill((255, 255, 255))
    for refLine in refLines:
    	refLine.draw(screen)
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
            	if event.button == 1:
            		if ongoingLine:
            			lines[-1].append(event.pos)
            			pygame.draw.line(screen, 0, lines[-1][-2], lines[-1][-1], 1)
            		else:
            			ongoingLine = True
            			lines.append([])
            			lines[-1].append(event.pos)
            	if event.button == 2 and ongoingLine:
            		ongoingLine = False
            		lines[-1].append(lines[-1][0])
            		pygame.draw.line(screen, 0, lines[-1][-2], lines[-1][-1], 1)
            	if event.button == 3 and ongoingLine:
            		ongoingLine = False

        pygame.display.update()
        clock.tick(60)
     
    for line in lines:
    	print('[')
    	for pt in line:
    		print(str(pt[0]) + " " + str(pt[1]))
    	print(']')
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()