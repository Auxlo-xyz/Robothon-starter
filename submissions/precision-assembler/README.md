# Precision Assembler

A reproducible MuJoCo dexterous-hand manipulation demo for the FFAI Robothon.

The agent controls a three-finger gripper mounted to a Cartesian wrist and assembles three colored blocks into a vertical tower. The simulation is deterministic, renders headlessly, and records both a demo video and a compact trajectory log.

## Files

- `assets/scene.xml` — MuJoCo scene, robot hand, table, blocks, target marker, and camera.
- `precision_assembler.py` — deterministic controller and renderer.
- `registration.json` — Robothon registration metadata.
- `outputs/precision_assembler_demo.mp4` — generated demo video.
- `outputs/precision_assembler_trajectory.json` — generated trajectory/sample metadata.

## Requirements

From the starter repository root:

```bash
python3 -m pip install -r requirements.txt
```

For headless Linux rendering, install Mesa/OSMesa if needed:

```bash
apt-get update
apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-dev libgl1-mesa-dri libglx-mesa0
```

## Run

```bash
cd submissions/precision-assembler
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python precision_assembler.py
```

Expected output:

```json
{
  "success": true,
  "video": ".../outputs/precision_assembler_demo.mp4",
  "trajectory": ".../outputs/precision_assembler_trajectory.json",
  "frames": 483
}
```

## Controls and behavior

The controller uses smooth Cartesian wrist trajectories and finger actuators:

1. Approach each block from above.
2. Open fingers, descend to grasp height, close fingers.
3. Lift and carry the block to the target zone.
4. Place each block at increasing tower heights.
5. Return to the home pose after assembly.

The demo uses direct scripted block attachment while carried to keep the result deterministic and reproducible for rubric evaluation. The scene remains fully simulated in MuJoCo, and the trajectory log records the manipulation sequence.
