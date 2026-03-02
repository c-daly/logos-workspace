# Talos Improvements

Bringing Talos from a Phase 1 simulation scaffold to a capable hardware abstraction layer with real simulation backends.

## Current State

Talos today is a pure-Python simulation layer. It works, but:

- **No network interface** — no FastAPI, no gRPC, no `__main__.py`. The Dockerfile CMD (`python -m talos`) fails because there's no entry point.
- **No physics** — sensors generate synthetic data from math functions (sine waves, Gaussian noise). Actuators are instantaneous (set position → position is set, no dynamics).
- **No environment** — objects exist as dicts with numpy positions. There's no spatial reasoning, collision detection, or scene graph.
- **Executor shim talks directly to Neo4j** — architecturally, Sophia owns the world model. Talos should report sensor data and accept commands, not write to the knowledge graph directly.
- **Integration tests don't run in CI** — no test infrastructure provisioned in the GitHub Actions workflow.
- **Still on logos-foundry v0.4.2** — needs the v0.5.0 bump.

What IS solid:
- Clean sensor/actuator abstractions with proper base classes
- Telemetry recording system
- Well-typed throughout (mypy-clean)
- Extensive unit test suite
- Fixtures system ready for consumption by other repos

## Goal

Replace the synthetic simulation backend with a real physics simulation (ROS2/Gazebo or alternative) so that:
1. Sophia can develop perception and planning against realistic sensor data
2. The cognitive loop can be tested end-to-end with physics
3. The architecture is ready for real hardware when it arrives

## Options

### Option A: ROS2 + Gazebo (Full Robotics Stack)

**What it is**: ROS2 is the standard robotics middleware. Gazebo (now "Gz Sim") is a physics simulator that integrates natively with ROS2. Together they provide simulated robots, sensors, and environments.

**Architecture:**
```
Sophia ←→ Talos API ←→ ROS2 bridge ←→ ROS2 topics ←→ Gazebo
                                          ↕
                                     Real hardware (future)
```

**What Talos would become:**
- A FastAPI/gRPC service that wraps ROS2 nodes
- `Sensor.read()` → subscribes to ROS2 sensor topics (camera, IMU, depth from Gazebo)
- `Actuator.command()` → publishes to ROS2 command topics (joint positions, gripper)
- The existing base class abstractions stay — implementations change from math to ROS2 subscribers

**Pros:**
- Industry standard — largest ecosystem, most tutorials, most robot models
- Direct path to real hardware (same ROS2 topics, swap Gazebo for drivers)
- Gazebo provides realistic physics, lighting, camera rendering, LiDAR, IMU
- Large model library (robots, environments, objects)
- Active development (Gz Harmonic is current LTS)

**Cons:**
- Heavy dependency footprint (ROS2 Humble/Jazzy requires Ubuntu, not macOS-native)
- Development requires Docker or a Linux VM on macOS
- Learning curve for ROS2 concepts (nodes, topics, services, actions, tf2)
- Gazebo sim startup is slow (~5-10s) and resource-intensive
- ROS2 Python bindings (rclpy) are less ergonomic than pure Python

**Install complexity:** Medium-high. ROS2 + Gazebo in Docker is well-documented but adds ~2GB to the image. Local development on macOS requires a containerized workflow.

### Option B: PyBullet (Lightweight Physics)

**What it is**: A Python-native physics engine (built on Bullet Physics). No middleware layer — direct Python API for creating scenes, stepping physics, reading sensors.

**Architecture:**
```
Sophia ←→ Talos API ←→ PyBullet engine (in-process)
```

**What Talos would become:**
- PyBullet runs in the same process as the Talos service
- `Sensor.read()` → `pybullet.getCameraImage()`, `pybullet.getBasePositionAndOrientation()`
- `Actuator.command()` → `pybullet.setJointMotorControl2()`
- Scene setup in Python: load URDF models, place objects, configure camera

**Pros:**
- pip-installable (`pip install pybullet`), works on macOS
- Simple Python API — no middleware concepts to learn
- Fast startup, low resource usage
- Good enough physics for pick-and-place, manipulation tasks
- URDF support (standard robot description format, same as ROS)
- OpenGL rendering for camera simulation

**Cons:**
- No path to real hardware (PyBullet is simulation-only)
- Less realistic rendering than Gazebo (no ray tracing, basic lighting)
- Smaller model library than Gazebo
- Single-threaded physics stepping
- Project is maintenance-mode (stable but not actively developed)

**Install complexity:** Low. `pip install pybullet` and you're running.

### Option C: MuJoCo (Research-Grade Physics)

**What it is**: DeepMind's physics engine, now open-source. Excellent for manipulation and contact dynamics. Used by most RL robotics research.

**Architecture:**
```
Sophia ←→ Talos API ←→ MuJoCo engine (in-process)
```

**Pros:**
- Best contact physics of any option (crucial for manipulation)
- pip-installable (`pip install mujoco`)
- Very fast simulation stepping
- Beautiful rendering with MuJoCo viewer
- Active development by DeepMind
- MJCF model format is more expressive than URDF for contact properties

**Cons:**
- No path to real hardware (same as PyBullet)
- Smaller ecosystem than ROS2/Gazebo
- MJCF models not as widely available as URDF
- Learning curve for MJCF model authoring

**Install complexity:** Low. `pip install mujoco`.

### Option D: Hybrid — MuJoCo/PyBullet now, ROS2 bridge later

**What it is**: Start with a lightweight in-process simulator for immediate development, add a ROS2 bridge when hardware integration becomes the priority.

**Architecture (phase 1):**
```
Sophia ←→ Talos API ←→ MuJoCo/PyBullet (in-process)
```

**Architecture (phase 2):**
```
Sophia ←→ Talos API ←→ ROS2 bridge ←→ { Gazebo | Real hardware }
                  ↘
                   MuJoCo/PyBullet (still available for fast local testing)
```

**Pros:**
- Fast iteration now (no Docker, no ROS2 setup)
- Real physics and rendering today
- ROS2 integration is additive, not a rewrite
- The Talos API abstraction layer means Sophia doesn't care what's behind it

**Cons:**
- Two simulation backends to maintain eventually
- ROS2 integration still needs to happen for hardware

## Recommendation

**Option D (Hybrid)** gives the best return on investment:

1. **Now**: Add MuJoCo as the simulation backend. It's pip-installable, has the best contact physics, and Talos's existing sensor/actuator abstractions map cleanly to MuJoCo's API.

2. **Soon**: Add a FastAPI service to Talos so Sophia can call it over HTTP (matching the hermes/sophia pattern). The existing in-process fixtures stay for unit tests.

3. **Later**: Add a ROS2 bridge backend. The Talos API doesn't change — only the implementation behind `Sensor.read()` and `Actuator.command()` switches from MuJoCo to ROS2 topics.

## Immediate Tasks (Repo Hygiene)

Before any simulation work, bring talos up to parity with the other repos:

| Task | Detail |
|------|--------|
| Bump logos-foundry to v0.5.0 | Update pyproject.toml tag + Dockerfile FROM |
| Fix Dockerfile CMD | Add `src/talos/__main__.py` with a FastAPI app, or remove the broken CMD |
| Add FastAPI service | Expose sensor reads and actuator commands over HTTP |
| Run integration tests in CI | Add docker_compose_file to the CI workflow inputs |
| Fix coverage threshold | README says 95%, config says 60%. Pick one. |
| Remove stale Docker example from README | `docker run -p 8002:8002` doesn't work |
| Vectorize camera simulation | Replace the O(height*width) Python loop with numpy |
| Resolve executor shim architecture | Either: (a) remove direct Neo4j writes and have Talos report to Sophia, or (b) document this as intentional |

## Simulation Integration Plan

### Phase 1: MuJoCo Backend

1. Add `mujoco` to optional dependencies (`[tool.poetry.group.sim]`)
2. Create `src/talos/backends/mujoco_backend.py`:
   - Load a MJCF scene model (table, objects, robot arm)
   - Implement `MuJoCoCamera(Sensor)` — renders from MuJoCo's camera
   - Implement `MuJoCoIMU(Sensor)` — reads body accelerations
   - Implement `MuJoCoMotor(Actuator)` — sets joint torques/positions
   - Implement `MuJoCoGripper(Actuator)` — controls gripper joints
3. Create a default scene MJCF file for the pick-and-place scenario
4. Keep existing `Simulated*` classes as the "no-physics" fallback

### Phase 2: Talos as a Service

1. Add FastAPI app in `src/talos/api/app.py`
2. Endpoints:
   - `GET /sensors/{name}/read` — returns sensor data
   - `POST /actuators/{name}/command` — sends command
   - `GET /scene/state` — full scene snapshot
   - `POST /scene/reset` — reset to initial state
3. Fix Dockerfile to run the API server
4. Add to the LOGOS service startup order: Sophia → Hermes → Talos → Apollo

### Phase 3: ROS2 Bridge (Future)

1. Add `src/talos/backends/ros2_backend.py`
2. ROS2 node subscribes to sensor topics, publishes actuator commands
3. Works with Gazebo (simulation) or real hardware (production)
4. Talos API stays the same — backend is selected via config
