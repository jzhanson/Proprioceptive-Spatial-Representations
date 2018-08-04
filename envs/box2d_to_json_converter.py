from __future__ import print_function

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

#import raptor_walker
import dog_walker

if __name__=='__main__':

    #env = raptor_walker.RaptorWalker()
    env = dog_walker.DogWalker()
    env.reset(STATIC=False)
    joints = env.joints
    bodies = [env.hull] + env.body + env.legs

    lower_limits = []
    upper_limits = []
    for i, jnt in enumerate(joints):
        lower_limits.append(jnt.lowerLimit)
        upper_limits.append(jnt.upperLimit)

    env.reset(STATIC=True)
    joints = env.joints
    bodies = [env.hull] + env.body + env.legs

    S = float(dog_walker.SCALE)

    # Take a few steps
    for i in range(200):
        env.render()
        env.step(np.zeros(env.action_space.shape))

    with open("box2d-json/DogWalker.json", "w") as jsonfile:
        print("{",file=jsonfile)

        # Write fixtures
        for i, b in enumerate(bodies):
            b._tmp_index = i
            b.user_data = i
            for j, f in enumerate(b.fixtures):
                vertices = [[v[0]*S,v[1]*S] for v in f.shape.vertices]
                print('\t"'+b.name+'.Fixture'+str(j)+'" : {', file=jsonfile)
                print('\t\t"DataType" : "Fixture",',file=jsonfile)
                print('\t\t"FixtureShape" : {',file=jsonfile)
                print('\t\t\t"Type" : "PolygonShape",',file=jsonfile)
                print('\t\t\t"Vertices" : '+str(vertices),file=jsonfile)
                print('\t\t},',file=jsonfile)
                print('\t\t"Friction" : '+str(f.friction)+',',file=jsonfile)
                print('\t\t"Density" : '+str(f.density)+',',file=jsonfile)
                print('\t\t"Restitution" : '+str(f.restitution)+',',file=jsonfile)
                print('\t\t"MaskBits" : '+str(f.filterData.maskBits)+',',file=jsonfile)
                print('\t\t"CategoryBits" : '+str(f.filterData.categoryBits),file=jsonfile)
                print('\t},', file=jsonfile)

        # Write bodies
        for i, b in enumerate(bodies):
            print('\t"'+b.name+'" : {',file=jsonfile)
            print('\t\t"DataType" : "DynamicBody",',file=jsonfile)
            print('\t\t"Position" : '+str([b.position.x*S, b.position.y*S])+',',file=jsonfile)
            print('\t\t"Angle" : '+str(b.angle)+',',file=jsonfile)
            print('\t\t"FixtureNames" : ["'+b.name+'.Fixture0"],',file=jsonfile)
            print('\t\t"Color1" : '+str(list(b.color1))+',',file=jsonfile)
            print('\t\t"Color2" : '+str(list(b.color2))+',',file=jsonfile)
            print('\t\t"CanTouchGround" : '+str(b.can_touch_ground).lower()+',',file=jsonfile)
            print('\t\t"InitialForceScale" : 0,',file=jsonfile)
            print('\t\t"Depth" : '+str(b.depth),file=jsonfile)
            print('\t},',file=jsonfile)

        # Write joints
        for i, jnt in enumerate(joints):
            bA = jnt.bodyA.userData
            bB = jnt.bodyB.userData
            # Convert anchors into local space
            anchorA = Box2D.b2Rot(-bA.angle) * (jnt.anchorA - bA.position)
            anchorB = Box2D.b2Rot(-bB.angle) * (jnt.anchorB - bB.position)
            anchorA = [anchorA[i]*S for i in range(2)]
            anchorB = [anchorB[i]*S for i in range(2)]
            a = bA._tmp_index
            b = bB._tmp_index
            #lowerLimit = jnt.lowerLimit
            #upperLimit = jnt.upperLimit
            lowerLimit = lower_limits[i]
            upperLimit = upper_limits[i]
            print('\t"Joint'+str(i)+'.'+bA.name+'.'+bB.name+'" : {',file=jsonfile)
            print('\t\t"DataType" : "JointMotor",',file=jsonfile)
            print('\t\t"BodyA" : "'+bA.name+'",',file=jsonfile)
            print('\t\t"BodyB" : "'+bB.name+'",',file=jsonfile)
            print('\t\t"LocalAnchorA" : '+str(anchorA)+',',file=jsonfile)
            print('\t\t"LocalAnchorB" : '+str(anchorB)+',',file=jsonfile)
            print('\t\t"EnableMotor" : '+str(jnt.motorEnabled).lower()+',',file=jsonfile)
            print('\t\t"EnableLimit" : '+str(jnt.limitEnabled).lower()+',',file=jsonfile)
            print('\t\t"MaxMotorTorque" : '+str(dog_walker.MOTORS_TORQUE)+',',file=jsonfile)
            print('\t\t"MotorSpeed" : '+str(jnt.motorSpeed)+',',file=jsonfile)
            print('\t\t"LowerAngle" : '+str(lowerLimit)+',',file=jsonfile)
            print('\t\t"UpperAngle" : '+str(upperLimit)+',',file=jsonfile)
            print('\t\t"Speed" : '+str(1)+',',file=jsonfile)
            print('\t\t"Depth" : '+str(jnt.depth),file=jsonfile)
            if i < len(joints)-1:
                print('\t},',file=jsonfile)
            else:
                print('\t}',file=jsonfile)

        print("}",file=jsonfile)
