# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keyboard viewer."""

import glfw
import imageio
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer


class KeyboardViewer(WindowViewer):  # pylint: disable=too-many-instance-attributes
    """Keyboard viewer."""

    def __init__(self, model, data, custom_key_press_callback) -> None:
        super().__init__(model, data)
        self._custom_key_press_callback = custom_key_press_callback

    def _create_overlay(self):
        """Create the on-screen menu overlay.

        Gymnasium's upstream implementation switches between `solver_iter` and
        `solver_niter` based on a string version comparison against MuJoCo.
        On Windows we have observed builds where that comparison selects the
        wrong field for the installed bindings, which crashes human rendering.
        Use attribute presence instead so the local Safety-Gym viewer remains
        compatible across MuJoCo package variants.
        """
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, "", "")
        else:
            self.add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self._run_speed,
                "[S]lower, [F]aster",
            )
        self.add_overlay(
            topleft, "Ren[d]er every frame", "On" if self._render_every_frame else "Off"
        )
        self.add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        self.add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        self.add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, "Stop", "[Space]")
            else:
                self.add_overlay(topleft, "Start", "[Space]")
                self.add_overlay(
                    topleft, "Advance simulation by one step", "[right arrow]"
                )
        self.add_overlay(
            topleft, "Referenc[e] frames", "On" if self.vopt.frame == 1 else "Off"
        )
        self.add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            self.add_overlay(topleft, "Cap[t]ure frame", "")
        self.add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        self.add_overlay(bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, ""))
        solver_iterations = None
        if hasattr(self.data, "solver_niter"):
            solver_niter = self.data.solver_niter
            solver_iterations = int(np.asarray(solver_niter).reshape(-1)[0]) + 1
        elif hasattr(self.data, "solver_iter"):
            solver_iterations = int(self.data.solver_iter) + 1
        if solver_iterations is not None:
            self.add_overlay(bottomleft, "Solver iterations", str(solver_iterations))
        self.add_overlay(
            bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep))
        )
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def _add_marker_to_scene(self, marker: dict):
        """Add a marker compatibly across MuJoCo Python binding variants.

        The Windows `thesis` env currently uses a Gymnasium renderer path that
        assumes legacy `MjvGeom` fields like `texid`. Some MuJoCo bindings no
        longer expose those fields, which crashes human rendering. Use
        `mjv_initGeom` and only assign optional attributes that actually exist.
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")

        geom = self.scn.geoms[self.scn.ngeom]
        geom_type = marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE)
        size = np.asarray(marker.get("size", np.array([0.01, 0.01, 0.01])), dtype=np.float64)
        pos = np.asarray(marker.get("pos", np.array([0.0, 0.0, 0.0])), dtype=np.float64)
        mat = np.asarray(marker.get("mat", np.eye(3).flatten()), dtype=np.float64)
        rgba = np.asarray(marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0])), dtype=np.float32)

        mujoco.mjv_initGeom(
            geom,
            geom_type,
            size=size,
            pos=pos,
            mat=mat,
            rgba=rgba,
        )

        if hasattr(geom, "category"):
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR

        for key, value in marker.items():
            if key in {"type", "size", "pos", "mat", "rgba"}:
                continue
            if key == "label":
                if hasattr(geom, "label"):
                    if value is None:
                        geom.label[0] = 0
                    else:
                        geom.label = value
                continue
            if not hasattr(geom, key):
                continue
            attr = getattr(geom, key)
            if isinstance(value, (tuple, list, np.ndarray)):
                arr = np.asarray(value)
                try:
                    attr[:] = arr.reshape(attr.shape)
                except Exception:
                    setattr(geom, key, value)
            else:
                setattr(geom, key, value)

        self.scn.ngeom += 1

    # pylint: disable-next=too-many-arguments, too-many-branches
    def _key_callback(self, window, key, scancode, action, mods):
        """Callback for keyboard events."""
        if action != glfw.RELEASE:
            pass
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # pylint: disable=no-member
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # pylint: disable=no-member
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (
                    glfw.get_framebuffer_size(self.window)[1],
                    glfw.get_framebuffer_size(self.window)[0],
                    3,
                ),
                dtype=np.uint8,
            )
            mujoco.mjr_readPixels(img, None, self.viewport, self.con)  # pylint: disable=no-member
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            # pylint: disable=no-member
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
            # pylint: enable=no-member
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        if key in (glfw.KEY_I, glfw.KEY_J, glfw.KEY_K, glfw.KEY_L):
            self._custom_key_press_callback(key=key, action=action)

        # Quit
        if key == glfw.KEY_ESCAPE:
            print('Pressed ESC')
            print('Quitting.')
            glfw.destroy_window(self.window)
            glfw.terminate()
