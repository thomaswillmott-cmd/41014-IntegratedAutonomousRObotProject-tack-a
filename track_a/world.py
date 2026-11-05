"""Assemble a room around the unmodified ROBOTIS TurtleBot3 model."""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START = np.array([-2.0, -1.5, 0.0])
MARKER_XY = np.array([2.82, 1.4])
MARKER_SIZE = 0.30  # Black outer square, excluding the white margin (metres).
CAMERA_OFFSET = 0.11
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.288
WAYPOINTS = np.array([[-1.4, -1.5], [-1.4, 1.4], [1.7, 1.4], [2.25, 1.4]])


def make_marker():
    path = ROOT / 'models' / 'marker_0.png'
    if not path.exists():
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image = np.full((600, 600), 255, np.uint8)
        image[50:550, 50:550] = cv2.aruco.generateImageMarker(dictionary, 0, 500)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError('Could not save marker texture')
    return path


def load_world(scenario=None):
    from .scenario import Scenario
    scenario = scenario or Scenario()
    source = ROOT / 'models' / 'robotis_tb3'
    root = ET.parse(source / 'turtlebot3_waffle_pi.xml').getroot()
    root.set('model', 'UTS 41014 - TurtleBot3 Track A')
    root.find('compiler').set('meshdir', str(source / 'assets'))
    root.find('option').set('timestep', '0.002')
    root.find('option').set('integrator', 'implicitfast')
    ET.SubElement(root, 'statistic', center='0 0 0.1', extent='6')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    ET.SubElement(visual, 'headlight', ambient='0.5 0.5 0.5', diffuse='0.6 0.6 0.6')
    assets = root.find('asset')
    ET.SubElement(assets, 'texture', name='tiles', type='2d', builtin='checker',
                  rgb1='0.18 0.23 0.28', rgb2='0.25 0.30 0.35', width='512', height='512')
    ET.SubElement(assets, 'material', name='floor_mat', texture='tiles', texrepeat='6 5', texuniform='true')
    ET.SubElement(assets, 'texture', name='marker_texture', type='2d', file=str(make_marker()))
    ET.SubElement(assets, 'material', name='marker_mat', texture='marker_texture',
                  texuniform='false', texrepeat='1 1', emission='0.6', specular='0')
    world = root.find('worldbody')
    base = world.find("body[@name='base']")
    base.set('pos', f'{scenario.start[0]} {scenario.start[1]} 0.005')
    base.set('euler', f'0 0 {scenario.start[2]}')
    # Keep ROBOTIS meshes for appearance, use smooth cylinders for wheel contact.
    # Low-friction rear supports must override the floor's contact properties.
    for geom in base.findall('geom'):
        if geom.get('type') == 'sphere':
            geom.set('priority', '1')
        elif geom.get('class') == 'collision' and geom.get('mesh') == 'waffle_pi_base':
            # Its convex mesh hull includes low rear details and otherwise drags
            # on the floor. Separate chassis contact from the caster supports.
            geom.attrib.clear()
            geom.attrib.update(name='chassis_collision', type='box',
                               pos='-0.06 0 0.05', size='0.14 0.14 0.025', group='3')
    wheel_defaults = root.find("default/default[@class='wheel']")
    wheel_defaults.find('joint').set('frictionloss', '0.005')
    wheel_defaults.find('joint').set('armature', '0.0001')
    wheel_defaults.find('velocity').set('kv', '0.3')
    for side in ('left', 'right'):
        wheel = base.find(f"body[@name='wheel_{side}']")
        collision = wheel.find("geom[@class='collision']")
        collision.attrib.clear()
        collision.attrib.update(name=f'{side}_wheel_collision', type='cylinder',
                                size='0.033 0.009', group='3', friction='1.2 0.005 0.0001')
    ET.SubElement(base, 'camera', name='robot_camera', pos=f'{CAMERA_OFFSET} 0 0.23',
                  xyaxes='0 -1 0 0 0 1', fovy='55')
    ET.SubElement(base, 'site', name='lidar', pos='-0.049 0 0.19', size='0.008', rgba='0 1 1 1')
    ET.SubElement(world, 'camera', name='overview', pos='0 -6 6', xyaxes='1 0 0 0 0.7071 0.7071')
    ET.SubElement(world, 'light', pos='0 0 4', directional='true', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 2.5 0.1', material='floor_mat')
    for name, pos, size in [
        ('north_wall', '0 2.55 0.3', '3.1 0.05 0.3'),
        ('south_wall', '0 -2.55 0.3', '3.1 0.05 0.3'),
        ('east_wall', '3.05 0 0.3', '0.05 2.5 0.3'),
        ('west_wall', '-3.05 0 0.3', '0.05 2.5 0.3'),
    ]:
        ET.SubElement(world, 'geom', name=name, type='box', pos=pos, size=size,
                      rgba='0.55 0.62 0.68 1' if 'wall' in name else '0.85 0.40 0.16 1')
    for name, kind, centre, halfsize in scenario.obstacles():
        size = f'{halfsize[0]} 0.25' if kind == 'cylinder' else f'{halfsize[0]} {halfsize[1]} 0.25'
        ET.SubElement(world, 'geom', name=name, type=kind, pos=f'{centre[0]} {centre[1]} 0.25',
                      size=size, rgba='0.85 0.40 0.16 1' if kind == 'box' else '0.2 0.55 0.75 1')
    if scenario.blocked:
        obstacle = ET.SubElement(world, 'body', name='surprise_obstacle', mocap='true', pos='0 0 -2')
        ET.SubElement(obstacle, 'geom', name='surprise_collision', type='cylinder', size='0.20 0.25', rgba='0.8 0.2 0.3 1')
    ET.SubElement(world, 'geom', name='dock_pad', type='box', pos='2.35 1.4 0.002',
                  size='0.4 0.3 0.002', rgba='0.15 0.75 0.5 1', contype='0', conaffinity='0')
    ET.SubElement(world, 'geom', name='dock_back', type='box', pos='2.85 1.4 0.23',
                  size='0.02 0.23 0.23', rgba='0.9 0.92 0.94 1')
    ET.SubElement(world, 'geom', name='dock_marker', type='plane', pos='2.82 1.4 0.25',
                  size='0.18 0.18 0.001', xyaxes='0 -1 0 0 0 1', material='marker_mat',
                  contype='0', conaffinity='0', group='2')
    from .scenario import STAGING, GOAL
    for i, point in enumerate([STAGING, GOAL[:2]]):
        ET.SubElement(world, 'site', name=f'waypoint_{i+1}', type='cylinder',
                      pos=f'{point[0]} {point[1]} 0.008', size='0.07 0.005', rgba='0.3 0.9 0.6 0.6')
    sensors = ET.SubElement(root, 'sensor')
    for side in ('left', 'right'):
        ET.SubElement(sensors, 'jointpos', name=f'encoder_{side}', joint=f'wheel_{side}')
        ET.SubElement(sensors, 'jointvel', name=f'speed_{side}', joint=f'wheel_{side}')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
