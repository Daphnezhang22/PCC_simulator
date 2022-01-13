from vpython import *

def Show_coordinator_xyz(x, y, z):
    coordinator_org = vec(x, y, z)
    arrow_X = arrow(pos=coordinator_org, axis=vector(0.5, 0, 0), shaftwidth=0.02,
                    color=vec(180 / 255, 180 / 255, 180 / 255))
    arrow_Y = arrow(pos=coordinator_org, axis=vector(0, 0.5, 0), shaftwidth=0.02,
                    color=vec(180 / 255, 180 / 255, 180 / 255))
    arrow_Z = arrow(pos=coordinator_org, axis=vector(0, 0, 0.5), shaftwidth=0.02,
                    color=vec(180 / 255, 180 / 255, 180 / 255))
    # text_x = text(pos=coordinator_org+vector(0.5, 0, 0), text='X', align='center', color=color.black)


def LightUp4():
    lamp41 = local_light(pos=vec(100, 100, -100), color=color.white * 0.4)
    lamp42 = local_light(pos=vec(100, -100, -100), color=color.white * 0.3)
    lamp41 = local_light(pos=vec(-100, 100, -100), color=color.white * 0.3)
    lamp42 = local_light(pos=vec(-100, -100, -100), color=color.white * 0.3)
    # lamp43 = local_light(pos=vec(-100,-100,100), color=color.white*0.2)
    # lamp44 = local_light(pos=vec(100,-100,100), color=color.white*0.2)
