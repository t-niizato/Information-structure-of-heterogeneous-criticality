# -*- coding: utf-8 -*-
# 群れの描画部分

# ■ 操作一覧
#	方向キー上下 or マウスホイール : ズーム
#	方向キー左右 or マウスドラッグ : カメラの軸回転
#	P : 一時停止
#	O : 1ステップだけ進む
#	T : 軌跡の表示切り替え
#   S : 再生スピード変更

import numpy as np
from vispy import gloo, app, visuals
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
import functions as fs
import math
import time


# ベースとなるモデル
class BaseModel():
    def __init__(self):
        return

    def draw_text(self, func, *args):
        return func(*args)


# 座標系のスケール
_P_SCALE = 0.05

# 軌跡の長さ
_CENTER_TRAIL_LENGTH = 1000

_model_position = np.array([[0, 0, 1], [0, 0.5, -1], [-1, 0, -1], [0, -0.5, -1], [1, 0, -1]])
_model_position *= 0.1
_cross_0_1_2 = np.cross(_model_position[1] - _model_position[0], _model_position[2] - _model_position[0])
_cross_0_1_4 = np.cross(_model_position[4] - _model_position[0], _model_position[1] - _model_position[0])
_cross_0_3_4 = np.cross(_model_position[3] - _model_position[0], _model_position[4] - _model_position[0])
_cross_0_2_3 = np.cross(_model_position[2] - _model_position[0], _model_position[3] - _model_position[0])
_cross_1_2_4 = np.cross(_model_position[1] - _model_position[4], _model_position[1] - _model_position[2])
_n = np.array([_cross_0_1_2, _cross_0_1_4, _cross_0_2_3, _cross_0_3_4, _cross_1_2_4])
_n_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4])
_n_indexed = _n[_n_idx]
_model_idx = np.array([0, 1, 2, 0, 1, 4, 0, 2, 3, 0, 3, 4, 1, 2, 4, 2, 3, 4])
_model_positions = _model_position[_model_idx]
_vtype = [('local_position', np.float32, 3),
          ('world_position', np.float32, 3),
          ('velocity', np.float32, 3),
          ('normal', np.float32, 3),
          ('color', np.float32, 3)]


def create_vertices(positions, velocities, colors):
    # Vertice colors

    """

    all_arr = [(_model_positions[i],positions[j],[0,0,0],[0,0,0]) for j in range(len(positions)) for i in range(len(_model_positions)) ]
    res = np.array(all_arr,dtype=_vtype)
    return res

    """

    model_positions_rep = np.tile(_model_positions.T, len(positions)).T
    world_positions_rep = np.repeat(positions, _model_positions.shape[0], 0)
    velocity_rep = np.repeat(velocities, _model_positions.shape[0], 0)
    colors_rep = np.repeat(colors, _model_positions.shape[0], 0)
    n_rep = np.tile(_n_indexed.T, len(positions)).T
    res = np.empty(model_positions_rep.shape[0], dtype=_vtype)
    res['local_position'] = model_positions_rep
    res['world_position'] = world_positions_rep
    res['velocity'] = velocity_rep
    res['normal'] = n_rep
    res['color'] = colors_rep

    """
    all_vertices = []
    for i in range(len(positions)):
        world_pos = positions[i]
        vel = velocities[i]
        model_vertices = [
        (_model_position[0], world_pos, vel, _n[0]), (_model_position[1], world_pos, vel, _n[0]), (_model_position[2], world_pos, vel, _n[0]), 
        (_model_position[0], world_pos, vel, _n[1]), (_model_position[1], world_pos, vel, _n[1]), (_model_position[4], world_pos, vel, _n[1]),
        (_model_position[0], world_pos, vel, _n[2]), (_model_position[2], world_pos, vel, _n[2]), (_model_position[3], world_pos, vel, _n[2]), 
        (_model_position[0], world_pos, vel, _n[3]), (_model_position[3], world_pos, vel, _n[3]), (_model_position[4], world_pos, vel, _n[3]),
        (_model_position[1], world_pos, vel, _n[4]), (_model_position[2], world_pos, vel, _n[4]), (_model_position[4], world_pos, vel, _n[4]), 
        (_model_position[2], world_pos, vel, _n[4]), (_model_position[3], world_pos, vel, _n[4]), (_model_position[4], world_pos, vel, _n[4])]
        #model_vertices = [([0,0,0], [0,0,0], [0,0,0], [0,0,0])]
        all_vertices.extend(model_vertices)
    res = np.array(all_vertices,dtype=_vtype)
    """

    return res


def create_indexes(n):
    indexes = []
    for i in range(n):
        arr = list(range(i * 18, i * 18 + 18))
        indexes.extend(arr)
    return indexes


def get_model_pos_and_vel(model, coding_style='niizato', dimension=3):
    if dimension == 2:
        positions = []
        velocities = []
        for i in range(len(model.positions)):
            pos = [model.positions[i][0], model.positions[i][1], 0.0]
            vel = [model.velocities[i][0], model.velocities[i][1], 0.0]
            positions.append(pos)
            velocities.append(vel)
        return positions, velocities

    if coding_style != 'niizato':
        return model.positions.copy(), model.velocities.copy()

    positions = []
    velocities = []
    for i in range(len(model.position[0])):
        pos = [model.position[0][i], model.position[1][i], model.position[2][i]]
        vel = [model.velocity[0][i], model.velocity[1][i], model.velocity[2][i]]
        positions.append(pos)
        velocities.append(vel)
    return positions, velocities


def get_model_color(model, coding_style='niizato'):
    if coding_style != 'niizato':
        return model.colors

    colors = []
    for i in range(len(model.color[0])):
        color = [model.color[0][i], model.color[1][i], model.color[2][i]]
        colors.append(color)
    return colors


vertex = """
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_ambient_color;


attribute vec3 local_position;
attribute vec3 normal;
attribute vec3 world_position;
attribute vec3 velocity;
attribute vec3 color;

varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;
varying vec4 v_ambient_color;
varying mat4x4 v_model;

mat4x4 lookAt(vec3 eye, vec3 center, vec3 up)
{
    vec3 f = normalize(center - eye);
    vec3 u = normalize(up);
    vec3 s = normalize(cross(f, u));
    u = cross(s, f);

    mat4x4 res;
    res[0][0] = s.x;
    res[1][0] = s.y;
    res[2][0] = s.z;
    res[3][0] =-dot(s, eye);

    res[0][1] = u.x;
    res[1][1] = u.y;
    res[2][1] = u.z;
    res[3][1] =-dot(u, eye);

    res[0][2] =-f.x;
    res[1][2] =-f.y;
    res[2][2] =-f.z;
    res[3][2] = dot(f, eye);

    res[0][3] = 0;
    res[1][3] = 0;
    res[2][3] = 0;
    res[3][3] = 0;

    return res;
}

void main()
{
    v_normal = normal;
    v_position = local_position;
    v_color = vec4(color,1.0);
    v_ambient_color = u_ambient_color;

    vec3 velocity_inv = velocity * -1;
    mat4x4 model = lookAt(vec3(0,0,0), velocity_inv, vec3(1,0,0));
    v_model = model;
    gl_Position = u_projection * u_view * (vec4(local_position,0.0)*model + vec4(world_position*0.05,1));
}
"""

fragment = """
uniform mat4 u_view;
uniform mat4 u_normal;

uniform vec3 u_light_intensity;
uniform vec3 u_light_position;
uniform float u_color_enable;

varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;
varying vec4 v_ambient_color;
varying mat4x4 v_model;

void main()
{
    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view * vec4(v_position, 1)*v_model);

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)
    // gl_FragColor = vec4(0.180,0.180,0.8,1) + (v_color * brightness * vec4(u_light_intensity, 1) + v_ambient_color)*0.32;
    gl_FragColor = v_color*u_color_enable + (1.0-u_color_enable)*vec4(0.180,0.180,0.8,1) + (v_color * brightness * vec4(u_light_intensity, 1) + v_ambient_color)*0.32;
}
"""

VERT_SHADER_LINE = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

FRAG_SHADER_LINE = """
void main()
{
    gl_FragColor = vec4(0.3,0.3,0.3,1);
}
"""


def lookAt(eye, target, up=[0, 0, 1]):
    """Computes matrix to put eye looking at target point."""
    eye = np.asarray(eye).astype(np.float32)  # [x,y,z]
    target = np.asarray(target).astype(np.float32)  # [x,y,z]
    up = np.asarray(up).astype(np.float32)

    vforward = eye - target
    vforward /= np.linalg.norm(vforward)
    vright = np.cross(up, vforward)
    vright /= np.linalg.norm(vright)
    vup = np.cross(vforward, vright)

    view = np.r_[vright, -np.dot(vright, eye),
                 vup, -np.dot(vup, eye),
                 vforward, -np.dot(vforward, eye),
                 [0, 0, 0, 1]].reshape(4, 4, order='F')

    return view


def distance(p1, p2):
    x_diff = p1[0] - p2[0]
    y_diff = p1[1] - p2[1]
    z_diff = p1[2] - p2[2]
    return math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)


class Drawer(app.Canvas):
    def __init__(self, model, dimension=3, coding_style='niizato', trail_enable=True, color_enable=False,
                 trail_length=5, trail_max_distance=500):

        if hasattr(model, "draw_text"):
            model.draw_text = self.draw_text

        self.coding_style = coding_style

        self.past_positions = []
        self.past_center_positions = []

        self.model = model
        self.N = 0
        if self.coding_style == 'niizato':
            self.N = len(self.model.position[0])
        else:
            self.N = len(self.model.positions)

        self.model_update_enable = True
        self.update_once = False
        self.zoom = 1.8
        self.eye_rad = 0.01
        self.mouse_pos_x = 0
        self.drag_on = False
        self.trail_enable = trail_enable
        self.color_enable = color_enable
        self.dimension = dimension

        self.trail_length = trail_length
        self.trail_max_distance = trail_max_distance
        self.sleep_time = 0.0

        app.Canvas.__init__(self, size=(700, 700), position=(610, 100), title='PyFlocking', keys='interactive')
        self.timer = app.Timer('auto', self.on_timer)

        positions, velocities = get_model_pos_and_vel(self.model, self.coding_style, self.dimension)
        V = None
        if self.color_enable == True:
            colors = get_model_color(self.model, coding_style=self.coding_style)
            V = create_vertices(positions, velocities, colors)
        else:
            colors = np.ones((self.N, 3))
            V = create_vertices(positions, velocities, colors)

        F = create_indexes(len(positions))
        self.vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)

        # Build view, model, projection & normal
        # --------------------------------------
        self.view = translate((-1, -3, -5))
        model = np.eye(4, dtype=np.float32)
        normal = np.array(np.matrix(np.dot(self.view, model)).I.T)

        # Build program
        # --------------------------------------
        self.program = Program(vertex, fragment)
        self.program.bind(self.vertices)
        # self.program['position'] = vertices
        self.program["u_light_position"] = 100, 100, 100
        self.program["u_light_intensity"] = 1, 1, 1
        self.program["u_view"] = self.view
        self.program["u_normal"] = normal
        if color_enable == True:
            self.program["u_color_enable"] = 1.0
        else:
            self.program["u_color_enable"] = 0.0

        # Text
        # --------------------------------------
        self.text = visuals.TextVisual('', bold=True, pos=(0., 0.))
        self.text.text = "text"
        self.text.anchors = ('left', 'center')
        self.text.font_size = 100
        self.text.pos = self.size[0] // 2, self.size[1] // 2
        # self.text.pos = [0,0]
        self.update()

        self.phi, self.theta = 0, 0
        self.dt = 0

        # --------------------------------------
        self.program_line = Program(VERT_SHADER_LINE, FRAG_SHADER_LINE)
        self.program_line['u_model'] = model
        self.program_line['u_view'] = self.view

        self.activate_zoom()

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=(0.95, 0.95, 0.95, 1.00), depth_test=True,
                       polygon_offset=(1, 1),
                       blend_func=('src_alpha', 'one_minus_src_alpha'),
                       line_width=0.75)
        self.timer.start()

        self.show()

    def on_draw(self, event):

        gloo.clear(color=True, depth=True)
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_ambient_color'] = 0.3, 0.1, 0.1, 0

        self.theta += .5
        self.phi += .5
        self.dt += 0.16

        if self.model_update_enable or self.model_update_once:
            if self.coding_style == 'niizato':
                self.model.updates()
            else:
                self.model.update()

            if self.trail_enable:
                positions, _ = get_model_pos_and_vel(self.model, self.coding_style, self.dimension)
                self.past_positions.append(positions)

            if len(self.past_positions) > self.trail_length:
                self.past_positions.pop(0)

            self.model_update_once = False

        if self.zoom < 0.01:
            self.zoom = 0.01

        center = [0.0, 0.0, 0.0]
        if self.coding_style == 'niizato' and self.dimension == 3:
            cx, cy, cz = fs.center_of_mass(self.model.position[0], self.model.position[1], self.model.position[2])
            center = [cx * _P_SCALE, cy * _P_SCALE, cz * _P_SCALE]

            if self.model_update_enable:
                self.past_center_positions.append(center)
                if len(self.past_center_positions) > _CENTER_TRAIL_LENGTH:
                    self.past_center_positions.pop(0)

        eye_base = [9.0001, 8.0001, 6.0001]
        eye_base[0] = math.sin(self.eye_rad) * 9.01
        eye_base[2] = math.cos(self.eye_rad) * 8.01

        if self.dimension == 2:
            eye_base = [0.0001, 0.0001, 6.0001]

        eye = [center[0] + eye_base[0] * self.zoom, center[1] + eye_base[1] * self.zoom,
               center[2] + eye_base[2] * self.zoom]
        self.view = lookAt(eye, center, [0, 1, 0])
        self.program["u_view"] = self.view
        self.program_line["u_view"] = self.view
        # self.program["u_view"] = lookAt([9.0001,8.0001,6.0001],[2,2,2],[0,1,0])

        # Draw model positions
        positions, velocities = get_model_pos_and_vel(self.model, self.coding_style, self.dimension)
        vertices = None
        if self.color_enable == True:
            colors = get_model_color(self.model, coding_style=self.coding_style)
            vertices = create_vertices(positions, velocities, colors)
        else:
            colors = np.ones((self.N, 3))
            vertices = create_vertices(positions, velocities, colors)
        # vertices = create_vertices(positions, velocities)
        # vertices = fs2.create_vertices(positions, velocities, _model_position, _n)
        vertices_buffer = VertexBuffer(vertices)
        self.program.bind(vertices_buffer)
        self.program.draw('triangles', self.faces)

        # ------------------------------------------------------
        # Draw line start
        # n = 6
        # a_position = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
        a_position = []

        # ------------------------------------------------------
        # Draw base grid
        if self.dimension == 3:
            grid_num = 10
            grid_scale = 10
            div = (grid_num / 2) * _P_SCALE * grid_scale
            for i in range(0, grid_num + 1):
                offs = i * _P_SCALE * grid_scale
                a_position.append([-div, 0, offs - div])
                a_position.append([div, 0, offs - div])
                a_position.append([offs - div, 0, -div])
                a_position.append([offs - div, 0, div])

        # ------------------------------------------------------
        # Draw Center Trail
        for i in range(len(self.past_center_positions) - 1):
            a_position.append(self.past_center_positions[i])
            a_position.append(self.past_center_positions[i + 1])
            pass

        # ------------------------------------------------------
        # Draw Trail
        for i in range(self.N):
            for j in range(len(self.past_positions) - 1):
                p1 = [self.past_positions[j][i][0] * _P_SCALE, self.past_positions[j][i][1] * _P_SCALE,
                      self.past_positions[j][i][2] * _P_SCALE]
                p2 = [self.past_positions[j + 1][i][0] * _P_SCALE, self.past_positions[j + 1][i][1] * _P_SCALE,
                      self.past_positions[j + 1][i][2] * _P_SCALE]
                if distance(p1, p2) < self.trail_max_distance * _P_SCALE:
                    a_position.append(p1)
                    a_position.append(p2)

        if len(a_position) > 0 and self.trail_enable:
            self.program_line['a_position'] = gloo.VertexBuffer(a_position)
            self.program_line.draw('lines')

        # Draw line end
        # ------------------------------------------------------

        # ------------------------------------------------------
        # play speed
        if self.sleep_time > 0.01:
            time.sleep(self.sleep_time)
        # ------------------------------------------------------

    def run(self):
        app.run()

    def on_key_press(self, event):
        key_name = event.key._names[0]
        if key_name is 'P' or key_name is 'Space':
            self.model_update_enable = not self.model_update_enable

        elif key_name is 'O':
            self.model_update_once = True
            self.model_update_enable = False

        if key_name is 'T':
            self.trail_enable = not self.trail_enable

        if key_name is 'S':
            self.sleep_time += 0.032
            if self.sleep_time > 0.14:
                self.sleep_time = 0.0


        elif key_name is 'Up':
            self.zoom -= 0.2
        elif key_name is 'Down':
            self.zoom += 0.2

        elif key_name is 'Right':
            self.eye_rad += 0.1
        elif key_name is 'Left':
            self.eye_rad -= 0.1

    def on_mouse_press(self, event):
        self.drag_on = True

    def on_mouse_release(self, event):
        self.drag_on = False

    def on_mouse_wheel(self, event):
        self.zoom -= event.delta[1] * 0.1

    def on_mouse_move(self, event):
        x, y = event.pos

        if self.mouse_pos_x != 0 and self.drag_on:
            mouse_diff = self.mouse_pos_x - x
            self.eye_rad += mouse_diff * 0.01

        self.mouse_pos_x = x

    def on_resize(self, event):
        self.phys_size = [event.physical_size[0], event.physical_size[1]]
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.text.transforms.configure(canvas=self, viewport=vp)
        self.activate_zoom()

    def draw_text(self, str, pos=[0, 0], size=30):
        self.text.text = str
        self.text.pos = pos
        self.text.font_size = size
        self.text.draw()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),
                                 0.1, 1000.0)
        self.program['u_projection'] = projection
        self.program_line['u_projection'] = projection

    def on_timer(self, event):
        self.update()