import os
import numpy as np
import pybullet as p

import env
from bodies.body import Body
from utils import get_quaternion


class Obj:
    def __init__(self, type=None, radius=0, half_extents=[0, 0, 0], length=0, normal=[0, 0, 1], filename='', scale=[1, 1, 1]):
        self.type = type
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.normal = normal
        self.filename = filename
        self.scale = scale

class Sphere(Obj):
    def __init__(self, radius=1):
        super().__init__(type=p.GEOM_SPHERE, radius=radius)

class Box(Obj):
    def __init__(self, half_extents=[1, 1, 1]):
        super().__init__(type=p.GEOM_BOX, half_extents=half_extents)

class Capsule(Obj):
    def __init__(self, radius=1, length=1):
        super().__init__(type=p.GEOM_CAPSULE, radius=radius, length=length)

class Cylinder(Obj):
    def __init__(self, radius=1, length=1):
        super().__init__(type=p.GEOM_CYLINDER, radius=radius, length=length)

class Plane(Obj):
    def __init__(self, normal=[0, 0, 1]):
        super().__init__(type=p.GEOM_PLANE, normal=normal)

class Mesh(Obj):
    def __init__(self, filename='', scale=[1, 1, 1]):
        super().__init__(type=p.GEOM_MESH, filename=filename, scale=scale)

def Shape(shape, static=False, mass=1.0, position=[0, 0, 0], orientation=[0, 0, 0, 1], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False, position_offset=[0, 0, 0], orientation_offset=[0, 0, 0, 1], local_env=None):
    local_env = local_env if local_env is not None else env.envir
    collision = p.createCollisionShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, height=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, collisionFramePosition=position_offset, collisionFrameOrientation=orientation_offset, physicsClientId=local_env.id) if collision else -1
    if rgba is not None:
        visual = p.createVisualShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, length=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, rgbaColor=rgba, visualFramePosition=position_offset, visualFrameOrientation=orientation_offset, physicsClientId=local_env.id) if visual else -1
    else:
        visual = p.createVisualShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, length=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, visualFramePosition=position_offset, visualFrameOrientation=orientation_offset, physicsClientId=local_env.id) if visual else -1
    if return_collision_visual:
        return collision, visual
    body = p.createMultiBody(baseMass=0 if static else mass, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=position, baseOrientation=get_quaternion(orientation), useMaximalCoordinates=maximal_coordinates, physicsClientId=local_env.id)
    return Body(body, local_env, collision_shape=collision, visual_shape=visual)

def Shapes(shape, static=False, mass=1.0, positions=[[0, 0, 0]], orientation=[0, 0, 0, 1], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False, position_offset=[0, 0, 0], orientation_offset=[0, 0, 0, 1], local_env=None):
    local_env = local_env if local_env is not None else env.envir
    collision = p.createCollisionShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, height=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, collisionFramePosition=position_offset, collisionFrameOrientation=orientation_offset, physicsClientId=local_env.id) if collision else -1
    if rgba is not None:
        visual = p.createVisualShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, length=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, rgbaColor=rgba, visualFramePosition=position_offset, visualFrameOrientation=orientation_offset, physicsClientId=local_env.id) if visual else -1
    else:
        visual = p.createVisualShape(shapeType=shape.type, radius=shape.radius, halfExtents=shape.half_extents, length=shape.length, fileName=shape.filename, meshScale=shape.scale, planeNormal=shape.normal, visualFramePosition=position_offset, visualFrameOrientation=orientation_offset, physicsClientId=local_env.id) if visual else -1
    if return_collision_visual:
        return collision, visual
    shape_ids = p.createMultiBody(baseMass=0 if static else mass, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=positions[0], baseOrientation=get_quaternion(orientation), batchPositions=positions, useMaximalCoordinates=maximal_coordinates, physicsClientId=local_env.id)
    shapes = []
    for body in shape_ids:
        shapes.append(Body(body, local_env, collision_shape=collision, visual_shape=visual))
    return shapes

def load_object(obj_name, obj_position, scale=[1.0, 1.0, 1.0], local_env=None):
    if obj_name == "apple_tree":
        obj_path = os.path.join(env.asset_dir, 'apple_tree', 'apple_tree.obj')
        object = Shape(Mesh(filename=obj_path, scale=scale), static=True, mass=50.0, position=obj_position, orientation=[0, 0, 0, 1], rgba=None, visual=True, collision=True, local_env=local_env)
    elif obj_name == "apple_tree_crook_canker":
        obj_path = os.path.join(env.asset_dir, 'apple_tree_crook_canker', 'apple_tree_crook_canker.obj')
        object = Shape(Mesh(filename=obj_path, scale=scale), static=True, mass=50.0, position=obj_position, orientation=[0, 0, 0, 1], rgba=None, visual=True, collision=True, local_env=local_env)
    else:
        # Load YCB objects
        obj_path = os.path.join(env.asset_dir, 'ycb', f'{obj_name}.obj')
        object = Shape(Mesh(filename=obj_path, scale=scale), static=False, mass=1.0, position=obj_position, orientation=[0, 0, 0, 1], rgba=None, visual=True, collision=True, local_env=local_env)
    return object

def URDF(filename, static=False, position=[0, 0, 0], orientation=[0, 0, 0, 1], maximal_coordinates=False, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    body = p.loadURDF(filename, basePosition=position, baseOrientation=get_quaternion(orientation), useMaximalCoordinates=maximal_coordinates, useFixedBase=static, physicsClientId=local_env.id)
    return Body(body, local_env)

# TODO: Should be in env? Problem is the get_quaternion function is in utils.py which depends on env.py (circular dependency)
def Ground(position=[0, 0, 0], orientation=[0, 0, 0, 1], local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return URDF(filename=os.path.join(env.asset_dir, 'plane', 'plane.urdf'), static=True, position=position, orientation=get_quaternion(orientation), local_env=local_env)
    # Randomly set friction of the ground
    # self.ground.set_frictions(self.ground.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)

def Line(start, end, radius=0.005, rgba=None, rgb=[1, 0, 0], replace_line=None, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    if rgba is None:
        rgba = list(rgb) + [1]
    # line = p.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end, lineColorRGB=rgba[:-1], lineWidth=1, lifeTime=0, physicsClientId=e.id)
    v1 = np.array([0, 0, 1])
    v2 = np.array(end) - start
    orientation = np.cross(v1, v2).tolist() + [np.sqrt((np.linalg.norm(v1)**2) * (np.linalg.norm(v2)**2)) + np.dot(v1, v2)]
    orientation = [0, 0, 0, 1] if np.linalg.norm(orientation) == 0 else orientation / np.linalg.norm(orientation)
    if replace_line is not None:
        replace_line.set_base_pos_orient(start + (np.array(end)-start)/2, orientation)
        return replace_line
    else:
        l = Shape(Cylinder(radius=radius, length=np.linalg.norm(np.array(end)-start)), static=True, position=start + (np.array(end)-start)/2, orientation=orientation, collision=False, rgba=rgba, local_env=local_env)
        local_env.visual_items.append(l)
        return l

def Points(point_positions, rgba=[1, 0, 0, 1], radius=0.01, replace_points=None, local_env=None):
    if type(point_positions[0]) not in (list, tuple, np.ndarray):
        point_positions = [point_positions]
    if replace_points is not None:
        for i in range(min(len(point_positions), len(replace_points))):
            replace_points[i].set_base_pos_orient(point_positions[i])
            return replace_points
    else:
        points = Shapes(Sphere(radius=radius), static=True, positions=point_positions, orientation=[0, 0, 0, 1], visual=True, collision=False, rgba=rgba, local_env=local_env)
        return points

# TODO: Move debug stuff to utils or env or something?
def DebugPoints(point_positions, points_rgb=[[1, 0, 0]], size=10, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    if type(point_positions[0]) not in (list, tuple, np.ndarray):
        point_positions = [point_positions]
    if type(points_rgb[0]) not in (list, tuple, np.ndarray):
        points_rgb = [points_rgb]*len(point_positions)
    debug_ids = []
    for i in range(len(point_positions)//4000 + 1):
        debug_id = -1
        while debug_id < 0:
            debug_id = p.addUserDebugPoints(pointPositions=point_positions[i*4000:(i+1)*4000], pointColorsRGB=points_rgb[i*4000:(i+1)*4000], pointSize=size, lifeTime=0, physicsClientId=local_env.id)
        debug_ids.append(debug_id)
    return debug_ids

def DebugLines(starts, ends, lines_rgb=[[1, 0, 0]], line_width=1.0, debug_ids=None, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    if type(starts[0]) not in (list, tuple, np.ndarray):
        starts = [starts]
    if type(ends[0]) not in (list, tuple, np.ndarray):
        ends = [ends]
    if type(lines_rgb[0]) not in (list, tuple, np.ndarray):
        lines_rgb = [lines_rgb]*len(starts)

    # Handle debug_ids properly: use provided list or create new list of Nones
    if debug_ids is None or len(debug_ids) == 0:
        debug_ids = [None]*len(starts)

    new_debug_ids = []
    for start_pt, end_pt, color, replace_id in zip(starts, ends, lines_rgb, debug_ids):
        debug_id = -1
        while debug_id < 0:
            if replace_id is None:
                debug_id = p.addUserDebugLine(lineFromXYZ=start_pt, lineToXYZ=end_pt, lineColorRGB=color, lineWidth=line_width, lifeTime=0, physicsClientId=local_env.id)
            else:
                debug_id = p.addUserDebugLine(lineFromXYZ=start_pt, lineToXYZ=end_pt, lineColorRGB=color, lineWidth=line_width, lifeTime=0, replaceItemUniqueId=replace_id, physicsClientId=local_env.id)
        new_debug_ids.append(debug_id)
    return new_debug_ids

def DebugCoordinateFrame(position=[0, 0, 0], orientation=[0, 0, 0, 1], axis_length=0.1, axis_radius=0.01, debug_ids=None, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    transform = lambda pos: p.multiplyTransforms(position, get_quaternion(orientation), pos, [0, 0, 0, 1], physicsClientId=local_env.id)[0]
    starts = [transform([0, 0, 0]), transform([0, 0, 0]), transform([0, 0, 0])]
    ends = [transform([axis_length, 0, 0]), transform([0, axis_length, 0]), transform([0, 0, axis_length])]
    lines_rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    debug_ids = DebugLines(starts, ends, lines_rgb=lines_rgb, line_width=axis_radius, debug_ids=debug_ids, local_env=local_env)
    return debug_ids

def clear_debug_items(debug_ids, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    for debug_id in debug_ids:
        p.removeUserDebugItem(debug_id, physicsClientId=local_env.id)

def visualize_coordinate_frame(position=[0, 0, 0], orientation=[0, 0, 0, 1], alpha=1.0, replace_old_cf=None, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    transform = lambda pos: p.multiplyTransforms(position, get_quaternion(orientation), pos, [0, 0, 0, 1], physicsClientId=local_env.id)[0]
    x = Line(start=transform([0, 0, 0]), end=transform([0.2, 0, 0]), rgba=[1, 0, 0, alpha], replace_line=None if replace_old_cf is None else replace_old_cf[0], local_env=local_env)
    y = Line(start=transform([0, 0, 0]), end=transform([0, 0.2, 0]), rgba=[0, 1, 0, alpha], replace_line=None if replace_old_cf is None else replace_old_cf[1], local_env=local_env)
    z = Line(start=transform([0, 0, 0]), end=transform([0, 0, 0.2]), rgba=[0, 0, 1, alpha], replace_line=None if replace_old_cf is None else replace_old_cf[2], local_env=local_env)
    return x, y, z

def clear_visual_item(items, local_env=None):
    if items is None:
        return
    local_env = local_env if local_env is not None else env.envir
    if type(items) in (list, tuple):
        for item in items:
            p.removeBody(item.body, local_env.id)
            # p.removeUserDebugItem(item, physicsClientId=local_env.id)
            for i in range(len(local_env.visual_items)):
                if local_env.visual_items[i] == item:
                    del local_env.visual_items[i]
                    break
    else:
        p.removeBody(items.body, local_env.id)
        # p.removeUserDebugItem(items, physicsClientId=local_env.id)
        for i in range(len(local_env.visual_items)):
            if local_env.visual_items[i] == items:
                del local_env.visual_items[i]
                break

def clear_all_visual_items(local_env=None):
    local_env = local_env if local_env is not None else env.envir
    for item in local_env.visual_items:
        p.removeBody(item.body, local_env.id)
    local_env.visual_items = []
    # p.removeAllUserDebugItems(physicsClientId=local_env.id) # Doesn't work

# TODO: Move to robot.py?
def salisbury_hand(finger_length=0.075, local_env=None):
    local_env = local_env if local_env is not None else env.envir

    length = finger_length

    link_c1, link_v1 = Shape(Box(half_extents=[0.1, 0.05, 0.01]), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, 0])
    link_c2, link_v2 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c3, link_v3 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c4, link_v4 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])

    link_c5, link_v5 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c6, link_v6 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c7, link_v7 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])

    link_c8, link_v8 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c9, link_v9 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])
    link_c10, link_v10 = Shape(Capsule(radius=0.01, length=length), visual=True, collision=True, rgba=[1, 1, 1, 1], return_collision_visual=True, position_offset=[0, 0, length/2])

    link_p1, link_o1 = [0, 0, 0], get_quaternion([0, 0, 0])
    link_p2, link_o2 = [-0.08, 0.02, 0.01], get_quaternion([0, 0, 0])
    link_p3, link_o3 = [0, 0, length], get_quaternion([0, 0, np.pi])
    link_p4, link_o4 = [0, 0, length], get_quaternion([0, 0, 0])

    link_p5, link_o5 = [0, -0.02, 0.01], get_quaternion([0, 0, 0])
    link_p8, link_o8 = [0.08, 0.02, 0.01], get_quaternion([0, 0, 0])

    linkMasses = [1]*9
    linkCollisionShapeIndices = [link_c2, link_c3, link_c4, link_c5, link_c6, link_c7, link_c8, link_c9, link_c10]
    linkVisualShapeIndices = [link_v2, link_v3, link_v4, link_v5, link_v6, link_v7, link_v8, link_v9, link_v10]
    linkPositions = [link_p2, link_p3, link_p4, link_p5, link_p3, link_p4, link_p8, link_p3, link_p4]
    linkOrientations = [link_o2, link_o3, link_o4, link_o5, link_o3, link_o4, link_o8, link_o3, link_o4]
    linkInertialFramePositions = [[0, 0, 0]]*9
    linkInertialFrameOrientations = [[0, 0, 0, 1]]*9
    linkParentIndices = [0, 1, 2, 0, 4, 5, 0, 7, 8]
    linkJointTypes = [p.JOINT_REVOLUTE]*9
    linkJointAxis =[[0, 0, 1], [1, 0, 0], [1, 0, 0]]*3

    multibody = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=link_c1, baseVisualShapeIndex=link_v1, basePosition=link_p1, 
        baseOrientation=link_o1, linkMasses=linkMasses, linkCollisionShapeIndices=linkCollisionShapeIndices, 
        linkVisualShapeIndices=linkVisualShapeIndices, linkPositions=linkPositions, linkOrientations=linkOrientations, 
        linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations, 
        linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis, physicsClientId=local_env.id
    )
    body = Body(multibody, local_env)
    body.controllable_joints = list(range(9))
    return body