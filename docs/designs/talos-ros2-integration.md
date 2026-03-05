# Talos: Hardware Abstraction and ROS2 Integration

## Context

- **Talos** is the LOGOS hardware abstraction layer (HAL). Its job is to make any robot look the same to Sophia.
- The user has a **TurtleBot3** running ROS2 with replacement LIDAR, IMU, odometry, and differential drive. No camera currently.
- Talos today has sensor ABCs (camera, depth, IMU) and actuator ABCs (motor, gripper) with simulated backends. No ROS2 support, no physics, no network interface.
- **Long-term vision:** Talos is a universal HAL. ROS2 is the first backend, not the only one. Any robot — ROS2, proprietary SDK, bare metal — plugs into Talos, and Sophia drives it the same way.

## Design Principles

1. **Talos is the abstraction, not ROS2.** ROS2 is a backend. The Talos ABCs define the contract. Sophia never knows or cares what's behind them.
2. **Start with ROS2, outgrow it.** ROS2 provides a mature ecosystem for the initial integration. Over time, Sophia takes over more capabilities. Eventually, ROS2 may be reduced to sensor transport.
3. **URDF as robot description.** Every ROS2 robot ships a URDF. Talos parses it, wraps it in a clean Python API, and populates the HCG with capability nodes. No manual manifest writing.
4. **Sophia owns the world model.** Talos reports sensor data and accepts commands. It does not write to the knowledge graph directly — that's Sophia's job.
5. **Safety is non-negotiable.** Reactive safety (e-stop, collision avoidance, velocity limits) runs at the Talos/ROS2 layer regardless of who's planning. Sophia can override navigation decisions but cannot override safety limits.
6. **Events, not polling.** Talos publishes meaningful state changes to the EventBus. Sophia subscribes. Not every sensor reading is an event — only significant changes.

## Architecture

```
Sophia (cognitive loop)
  Goals, plans, reasoning over HCG entities
      ↕ EventBus + API
Talos HAL
  Sensor/actuator ABCs, robot description, capability manifest
      ↕ Backend interface
Backend adapters (pluggable)
  ├── ROS2Backend (first target)
  ├── SimulatedBackend (current, no physics)
  └── future: proprietary SDKs, bare metal, etc.
      ↕
Hardware / Simulator
  ├── Gazebo (simulated TurtleBot3)
  └── Real TurtleBot3
```

### Sensor ABCs

Existing: `Camera`, `DepthSensor`, `IMU`

New ABCs needed:
- **`LidarSensor`** — returns scan data (ranges, angles, intensities)
- **`OdometrySensor`** — returns pose (position + orientation) and twist (linear + angular velocity)

Each ABC has a `read() -> TypedData` method. Backends implement the ABC with their transport (ROS2 topics, simulation math, raw serial, etc.)

### Actuator ABCs

Existing: `Motor`, `Gripper`

New ABCs needed:
- **`DifferentialDrive`** — accepts linear + angular velocity (maps to `/cmd_vel` in ROS2)

### ROS2 Backend

| Talos ABC | ROS2 Topic | Message Type |
|-----------|-----------|--------------|
| `ROS2Lidar` | `/scan` | `sensor_msgs/LaserScan` |
| `ROS2IMU` | `/imu` | `sensor_msgs/Imu` |
| `ROS2Odometry` | `/odom` | `nav_msgs/Odometry` |
| `ROS2DiffDrive` | `/cmd_vel` | `geometry_msgs/Twist` |
| `ROS2Camera` (future) | `/camera/image_raw` | `sensor_msgs/Image` |

Nav2 integration: Talos sends navigation goals to nav2's action server. Nav2 handles SLAM, path planning, obstacle avoidance. Sophia monitors progress through EventBus events.

`rclpy` is an optional dependency (`poetry install --extras ros2`).

### Robot Description (URDF)

Every ROS2 robot has a URDF describing its links, joints, sensors, and physical properties. Talos:

1. **Parses the URDF** using an existing Python parser (e.g., `urdf_parser_py` or `yourdfpy`)
2. **Wraps it in a clean Python API** — no one touches XML directly
3. **Populates HCG capability nodes** — Sophia knows "I have a LIDAR with 360° FOV and 12m range at position (0, 0, 0.15)" because the URDF said so

This means any ROS2 robot with a URDF is automatically described to Sophia.

### Event Integration

Talos translates significant state changes into EventBus events:

| Event | Trigger |
|-------|---------|
| `obstacle_detected` | New obstacle in LIDAR scan within threshold |
| `obstacle_cleared` | Previously detected obstacle gone |
| `nav_goal_reached` | Nav2 reports goal success |
| `nav_goal_failed` | Nav2 reports failure or timeout |
| `area_mapped` | SLAM adds significant new map data |
| `battery_low` | Battery below threshold |
| `sensor_fault` | Sensor stops responding or returns invalid data |

### Spatial Knowledge

- SLAM map data feeds into the HCG as spatial entities (locations, obstacles, paths, rooms)
- Sophia queries the graph: "what's near me?", "is the path to X clear?"
- Over time, Sophia annotates spatial knowledge with learned information ("this hallway is congested at 3pm")

### Talos as a Service

Following the hermes/sophia pattern, Talos exposes a FastAPI service:

- `GET /sensors/{name}/read` — current sensor data
- `POST /actuators/{name}/command` — send command
- `GET /robot/capabilities` — full capability manifest from URDF
- `GET /robot/state` — current pose, sensor summary
- `POST /navigation/goal` — send nav2 goal
- `GET /health` — standard LOGOS health response

## Migration Path: ROS2 → Sophia Control

Each capability sits on a spectrum. The architecture supports moving along this spectrum without rewrites.

### Phase 1: ROS2 Does Everything
- Sophia sets high-level goals ("explore room", "go to kitchen")
- Talos translates to nav2 goals
- ROS2 handles SLAM, path planning, obstacle avoidance
- Sophia observes outcomes through EventBus
- Spatial knowledge accumulates in HCG

### Phase 2: Sophia Monitors and Learns
- Sophia correlates actions with outcomes, builds causal knowledge in HCG
- "Fast turn on smooth floor → drift", "this distance at this speed → collision"
- Feedback dispatcher records navigation outcomes
- Maintenance scheduler triggers reflection on spatial knowledge

### Phase 3: Sophia Starts Overriding
- Sophia makes better decisions than nav2 using context nav2 lacks
- "Don't take that route — it was blocked yesterday"
- Talos supports override: Sophia can intercept/modify nav2 plans
- Nav2 remains the fallback

### Phase 4: Sophia Drives Directly
- Sophia issues motor commands through Talos actuator ABCs
- ROS2 reduced to sensor transport and safety limits
- Requires the grounding/physical knowledge goal to be mature
- Sophia has learned enough physics to drive safely

## TurtleBot3 Specifics

| Property | Value |
|----------|-------|
| Platform | TurtleBot3 |
| ROS2 | Yes |
| LIDAR | Replacement unit (model TBD) |
| IMU | Built-in |
| Drive | Differential (2-wheel + caster) |
| Camera | None currently (future addition) |
| Max speed | ~0.22-0.26 m/s depending on model |

Adding a camera later activates the visual perception pipeline and connects to the grounding/physical knowledge goal.

### Gazebo First

Development and testing happens against a **Gazebo simulation** of the TurtleBot3. Same ROS2 topics, same nav2 stack, zero risk. Switching to the real robot is a launch file change — no code changes.

## What Needs to Be Built

### Talos
1. `LidarSensor` and `OdometrySensor` ABCs + simulated implementations
2. `DifferentialDrive` actuator ABC + simulated implementation
3. ROS2 backend adapter (`rclpy` optional dependency)
4. URDF parser + Python facade for robot description
5. HCG capability node population from URDF
6. FastAPI service layer (sensor reads, actuator commands, nav goals)
7. EventBus integration for state change events
8. Backend registry — select backend via config

### Sophia
1. Spatial knowledge service — writes sensor-derived spatial data to HCG
2. Navigation goal interface — translates cognitive goals to Talos commands
3. Observation/learning pipeline — correlates actions with outcomes
4. Capability reasoning — plans against robot's actual capabilities from HCG

### Logos (Foundry)
1. Spatial entity types in the ontology (Location, Obstacle, Path, Room)
2. Robot capability types (Sensor, Actuator, with properties from URDF)
3. Spatial edge types (NEAR, BLOCKS, CONNECTS, NAVIGABLE_TO)

## Universality

ROS2 is the starting point, not the destination. The Talos HAL is designed to abstract over any robot platform:

- **ROS2 robots** — use the ROS2 backend, URDF for description
- **Proprietary SDK robots** — new backend implementing the same ABCs, manual or SDK-parsed capability description
- **Bare metal / microcontroller** — serial or network backend, minimal capability description
- **Cloud robotics / remote** — network backend with latency-aware command buffering

Think of it like Apollo: Apollo works as a command-line interface, but can expand into a touchscreen with microphone and speaker. The cognitive layer is the same — the interface changes. Talos is the same idea for the physical world. Eventually, Sophia reasons the same way whether she is driving a TurtleBot3 through ROS2, a robotic arm through a proprietary SDK, or a simulated agent through Gazebo. The HAL absorbs the differences.

The contract is always the same: Talos ABCs for sensors and actuators, capability manifest in the HCG, EventBus for state changes. Sophia doesn't know or care what's behind the abstraction.

## Open Questions

1. Which TurtleBot3 model (Burger vs Waffle)?
2. What LIDAR replacement was used? (affects FOV, range, scan rate)
3. Should Talos run in the same process as Sophia, or as a separate service?
4. Minimum viable demo — "Sophia commands explore, builds spatial map in HCG"?
5. How to handle ROS2 dependency in CI? (Docker with ROS2 image, or mock the backend?)
