import pygame, random
pygame.init()
surf = pygame.display.set_mode((400,400))


names = []

f_nam = open('names.dat', 'r')
s = f_nam.read().split('\n')
for _ in s:
    names.append(_.split(',')[0])

f_nam.close()

data = open('colors.dat', 'a')
name = open('names.dat', 'a')



for _ in range(32):
    pygame.event.get()
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    surf.fill((r,g,b))
    pygame.display.flip()
    s = input('%d, %d, %d>' % (r,g,b))
    i = 0
    for _ in names:
        if _ == s:
            s = None
            break
        i += 1
    if s != None:
        print('append', s)
        names.append(s)
        name.write('%s, %d\n' % (s, i))
    data.write('%d,%d,%d:%d\n'%(r, g, b, i))
        

pygame.quit()
data.close()
name.close()
