import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE / "assets" / "scene.xml"
OUTDIR = HERE / "outputs"
VIDEO = OUTDIR / "precision_assembler_demo.mp4"
TRAJ = OUTDIR / "precision_assembler_trajectory.json"
FPS = 30
WIDTH = 960
HEIGHT = 540

OPEN_HAND = np.array([0.05, 0.15, 0.05, 0.15, 0.05, 0.15])
CLOSED_HAND = np.array([0.95, 1.05, 0.95, 1.10, 0.95, 1.10])

BLOCKS = [
    ("block_red", np.array([0.34, 0.16, 0.31]), np.array([0.45, 0.00, 0.31])),
    ("block_green", np.array([0.52, -0.16, 0.31]), np.array([0.45, 0.00, 0.38])),
    ("block_blue", np.array([0.64, 0.08, 0.31]), np.array([0.45, 0.00, 0.45])),
]


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def set_pose(data, model, xyz, yaw=0.0, hand=OPEN_HAND):
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")] = xyz[0]
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")] = xyz[1]
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z")] = xyz[2] - 0.58
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw")] = yaw
    for name, value in zip(["thumb_base", "thumb_tip", "index_base", "index_tip", "middle_base", "middle_tip"], hand):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = float(value)


def block_qpos_addr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[jid]


def set_block_pose(data, model, joint_name, pos):
    adr = block_qpos_addr(model, joint_name)
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]:][:6] = 0


def run_segment(model, data, renderer, frames, samples, start, end, seconds, yaw=0.0, hand0=OPEN_HAND, hand1=OPEN_HAND, carry=None):
    steps = max(1, int(seconds * FPS))
    substeps = max(1, int(round((1.0 / FPS) / model.opt.timestep)))
    for i in range(steps):
        a = smoothstep(i / max(1, steps - 1))
        xyz = start * (1 - a) + end * a
        hand = hand0 * (1 - a) + hand1 * a
        set_pose(data, model, xyz, yaw=yaw, hand=hand)
        if carry:
            joint_name, offset = carry
            set_block_pose(data, model, joint_name, xyz + offset)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="demo")
        frames.append(renderer.render())
        if len(samples) % 10 == 0:
            samples.append({"frame": len(frames), "palm_target_xyz": np.round(xyz, 4).tolist(), "carrying": carry[0] if carry else None})
        else:
            samples.append({"frame": len(frames), "carrying": carry[0] if carry else None})


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    mujoco.mj_resetData(model, data)

    for name, start, _ in BLOCKS:
        set_block_pose(data, model, name.replace("block_", "") + "_free", start)

    frames = []
    samples = []
    home = np.array([0.28, 0.0, 0.56])
    set_pose(data, model, home)
    for _ in range(90):
        mujoco.mj_step(model, data)

    current = home.copy()
    for block_name, start_pos, target_pos in BLOCKS:
        joint_name = block_name.replace("block_", "") + "_free"
        above_pick = start_pos + np.array([0, 0, 0.20])
        grasp = start_pos + np.array([0, 0, 0.105])
        above_place = target_pos + np.array([0, 0, 0.20])
        place = target_pos + np.array([0, 0, 0.105])
        carry_offset = np.array([0.0, 0.0, -0.105])

        run_segment(model, data, renderer, frames, samples, current, above_pick, 0.8, hand0=OPEN_HAND, hand1=OPEN_HAND)
        run_segment(model, data, renderer, frames, samples, above_pick, grasp, 0.65, hand0=OPEN_HAND, hand1=OPEN_HAND)
        run_segment(model, data, renderer, frames, samples, grasp, grasp, 0.45, hand0=OPEN_HAND, hand1=CLOSED_HAND)
        run_segment(model, data, renderer, frames, samples, grasp, above_pick, 0.65, hand0=CLOSED_HAND, hand1=CLOSED_HAND, carry=(joint_name, carry_offset))
        run_segment(model, data, renderer, frames, samples, above_pick, above_place, 0.95, hand0=CLOSED_HAND, hand1=CLOSED_HAND, carry=(joint_name, carry_offset))
        run_segment(model, data, renderer, frames, samples, above_place, place, 0.65, hand0=CLOSED_HAND, hand1=CLOSED_HAND, carry=(joint_name, carry_offset))
        set_block_pose(data, model, joint_name, target_pos)
        run_segment(model, data, renderer, frames, samples, place, place, 0.45, hand0=CLOSED_HAND, hand1=OPEN_HAND)
        run_segment(model, data, renderer, frames, samples, place, above_place, 0.55, hand0=OPEN_HAND, hand1=OPEN_HAND)
        current = above_place

    run_segment(model, data, renderer, frames, samples, current, home, 1.0, hand0=OPEN_HAND, hand1=OPEN_HAND)
    imageio.mimsave(str(VIDEO), frames, fps=FPS, macro_block_size=16)
    TRAJ.write_text(json.dumps({"scene": str(SCENE), "video": str(VIDEO), "frames": len(frames), "fps": FPS, "samples": samples[:80]}, indent=2))
    print(json.dumps({"success": True, "video": str(VIDEO), "trajectory": str(TRAJ), "frames": len(frames)}, indent=2))


if __name__ == "__main__":
    main()
