import os
import pybullet as p
import pybullet_data

def create_env():
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -0.1)
    pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),basePosition=[0.5,0,-0.65])
    # trayUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "tray/traybox.urdf"),basePosition=[0.65,0,0])
    boxUid = p.loadURDF("./urdf/square_20mm.urdf",basePosition=[0.3,0.3,0])
    objectSize = [0.01, 0.01, 0.1]
    objectUidC = p.createCollisionShape(p.GEOM_BOX, halfExtents=objectSize)
    objectUidV = p.createCollisionShape(p.GEOM_BOX, halfExtents=objectSize)
    objectUid = p.createMultiBody(0.1, objectUidC, objectUidV , basePosition = [0.59, -0.02, 1])
    while True:
        p.stepSimulation()

if __name__ == '__main__':
    create_env()
