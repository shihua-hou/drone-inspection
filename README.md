# 园区无人机巡检仿真

ROS 2 Humble + PX4 SITL + Gazebo Harmonic（gz-sim 8）的单机三维园区巡检仿真。飞机按已知障碍物地图规划一条楼间航线，用 MAVROS Offboard 位置 setpoint 飞完 5 个巡检点后返航降落。深度相机（可选 16 线雷达）同时建一张在线占据图，用来做地图对比；前沿探索和在线重规划是实验功能，默认关闭。

> **仅用于仿真与算法开发，不是实飞软件。** 没有在线 SLAM，也没有实飞安全认证。未知体素按可通行处理。默认飞行始终使用已知地图。

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [仓库结构](#仓库结构)
- [依赖](#依赖)
- [编译](#编译)
- [运行仿真](#运行仿真)
- [Launch 参数](#launch-参数)
- [坐标系](#坐标系)
- [园区场景](#园区场景)
- [规划与飞行](#规划与飞行)
- [建图与地图对比](#建图与地图对比)
- [实验功能](#实验功能)
- [话题与输出文件](#话题与输出文件)
- [配置](#配置)
- [测试](#测试)
- [常见问题](#常见问题)
- [后续方向](#后续方向)
- [许可证](#许可证)

## 功能概览

- **Gazebo 园区**：围墙内约 230 m × 190 m。内园是规整街区（30 栋楼，高 7–30 m，立面间距至少 8 m），外圈有停车场、足球场、体育馆、食堂、仓库、访客中心和树林公园，最外层是铁艺围栏和南大门。
- **物理与光照**：每个 link 都有碰撞和惯性，重力开启。草地有 0–0.3 m 起伏并做网格碰撞；道路、人行道、车、灯、牌、绿篱、围墙都有碰撞。一个开阴影的方向光，带天空和云。轻风约 2 m/s，PX4 螺旋桨模型会把它算进空速。
- **默认航线**：5 个巡检点都在楼缝里，整条航线约 354 m，最高不超过 12 m，低于内园大多数楼顶。
- **3D A\***：26 连通栅格，障碍膨胀 2.5 m，视线平滑。对膨胀后的障碍盒做精确线段相交检测。
- **Offboard 执行**：20 Hz 本地位置 setpoint。起飞前先算完整条路线，避免规划卡住控制周期。飞完后返回起点上方并请求 `AUTO.LAND`。
- **在线建图**：默认用前视深度相机；可选机顶 16 线雷达（按 VLP-16：水平 360°、垂直 ±15°、最远 100 m，5 Hz，Gazebo 里不画射线）。对数几率占据，OctoMap 风格的 DDA 射线更新。
- **地图对比**：先验地图、仅传感器地图、两者合并三条路线的航程，以及已观测区域的精度/召回。后台线程、最多每 10 秒一次。
- **可选前沿探索**：参考 [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL)，默认关闭，只发布可视化，不替换 Offboard 指令。
- **可选在线重规划**：默认关闭。未知体素仍按可通行处理。不是经验证的避碰系统。

闭环试飞（无界面 Gazebo + PX4 `x500_depth` + MAVROS）曾跑通：5/5 巡检点，航程约 359 m，用时约 59 s，离障碍最近 2.68 m，setpoint 最大间隔 51 ms。

## 系统架构

```text
Gazebo park.sdf ──深度点云──► ros_gz_bridge ──► online_mapper
       ▲                         │                    │
       │                    可选雷达点云               │ 在线占据快照
PX4 SITL x500                    │                    ▼
       │                    MAVROS ◄──────── inspection_mission
       └──── 20 Hz Offboard 位置 setpoint ────────────┘
```

三个进程必须按顺序起来：

1. `gz sim` 加载园区世界（PX4 自己再挂物理和传感器插件）。
2. PX4 SITL 把飞机生成到已经在跑的 `park` 世界。
3. `ros2 launch` 启动 MAVROS、任务节点、建图节点和 RViz。

`park_inspection/campus.py` 是园区的唯一数据源。生成器写出 `worlds/park.sdf`、`worlds/meshes/` 和 `config/campus_obstacles.yaml`。规划器、Gazebo 碰撞和测试共用同一套包围盒。

## 仓库结构

```text
drone-inspection/
├── README.md
├── LICENSE
└── src/park_inspection/          ROS 2 包
    ├── park_inspection/          Python 模块
    │   ├── campus.py             园区布局（唯一数据源）
    │   ├── planning.py           3D A*
    │   ├── occupancy.py          对数几率占据与路线对比
    │   ├── mapping_node.py       建图节点 online_mapper
    │   ├── mission_node.py       任务节点 inspection_mission
    │   ├── exploration.py        FUEL 风格前沿（默认关）
    │   └── sensing.py            传感器外参
    ├── tools/
    │   ├── generate_campus.py    写 SDF / YAML / 网格
    │   └── meshgen.py            OBJ 与程序贴图
    ├── worlds/                   生成文件：park.sdf + meshes/
    ├── config/
    │   ├── park_mvp.yaml         任务与传感器参数
    │   ├── campus_obstacles.yaml 生成的障碍、巡检点、边界
    │   ├── gz_points_bridge.yaml
    │   └── gz_lidar_bridge.yaml
    ├── models/x500_depth_lidar/  带 16 线雷达的机型
    ├── launch/inspection.launch.py
    ├── rviz/inspection.rviz
    └── test/                     pytest
```

`worlds/park.sdf`、`worlds/meshes/` 和 `config/campus_obstacles.yaml` 都是生成文件，不要手改。

## 依赖

在 **Ubuntu 22.04** 上：

| 组件 | 说明 |
| --- | --- |
| ROS 2 Humble | `rclpy`、`mavros`、`mavros_msgs`、`ros-humble-ros-gz-bridge`、`rviz2` |
| Gazebo Harmonic | gz-sim 8，带 ogre2 渲染 |
| PX4 SITL | 与 Gazebo Harmonic 匹配的源码树，已编译 `px4_sitl` |
| Python | numpy、PyYAML、Pillow（生成贴图时需要） |

PX4 建议放在 `~/PX4-Autopilot`，并已执行过一次 `make px4_sitl gz_x500_depth`，这样才有 `build/px4_sitl_default/rootfs/gz_env.sh`。

系统包示例：

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-humble-mavros ros-humble-mavros-msgs \
  ros-humble-ros-gz-bridge python3-numpy python3-yaml python3-pil python3-pytest
```

GeographicLib 数据集是 MAVROS 需要的，按 MAVROS 文档安装即可。

## 编译

```bash
source /opt/ros/humble/setup.bash
cd ~/drone-inspection
colcon build --symlink-install
source install/setup.bash
```

只重编本包：

```bash
colcon build --symlink-install --packages-select park_inspection
```

## 运行仿真

开三个终端，按 1 → 2 → 3 的顺序。改过代码或第一次运行先编译。

### 1. 园区世界

```bash
source /opt/ros/humble/setup.bash
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
gz sim -r ~/drone-inspection/src/park_inspection/worlds/park.sdf
```

必须先 `source gz_env.sh`，否则深度相机的传感器插件加载不了。世界用相对路径引用 `worlds/meshes/`，首次加载要几秒。请直接打开源码目录或安装目录里的 `park.sdf`，不要单独把它拷走。

无界面时把 `-r` 换成 `-s -r --headless-rendering`。

### 2. PX4 SITL

起飞坪比路面高 9 cm。`PX4_GZ_MODEL_POSE="0,0,0.35"` 让飞机从坪上方落下，不要生成在碰撞体里。

**A. 只带深度相机（默认）**

```bash
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=park PX4_SYS_AUTOSTART=4002 \
PX4_GZ_MODEL_POSE="0,0,0.35" \
PX4_PARAM_NAV_DLL_ACT=0 PX4_PARAM_CBRK_SUPPLY_CHK=894281 \
HEADLESS=1 make px4_sitl gz_x500_depth
```

**B. 深度相机 + 16 线雷达**

`make px4_sitl gz_x500_depth` 会把机型名固定成不带雷达的模型，所以这里直接跑 SITL 二进制。`PX4_GZ_MODELS` 指到本仓库后，Gazebo 才能读到 `x500_depth_lidar`；`x500_depth` 本体仍从 PX4 的模型路径加载。

```bash
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
export GZ_SIM_RESOURCE_PATH="$HOME/drone-inspection/src/park_inspection/models:${GZ_SIM_RESOURCE_PATH}"
export PX4_GZ_MODELS="$HOME/drone-inspection/src/park_inspection/models"
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=park PX4_SYS_AUTOSTART=4002 \
PX4_SIM_MODEL=gz_x500_depth_lidar PX4_GZ_MODEL_POSE="0,0,0.35" \
PX4_PARAM_NAV_DLL_ACT=0 PX4_PARAM_CBRK_SUPPLY_CHK=894281 \
build/px4_sitl_default/bin/px4
```

`NAV_DLL_ACT=0` 和 `CBRK_SUPPLY_CHK=894281` **只给本地 SITL 用**：没有 GCS 心跳也能解锁，并跳过硬件供电检查。不要写到真机上。

环境变量有时不会生效。PX4 起来后务必在 `pxh>` 里确认：

```text
param set NAV_DLL_ACT 0
param set CBRK_SUPPLY_CHK 894281
param show NAV_DLL_ACT
param show CBRK_SUPPLY_CHK
```

两个值必须是 `0` 和 `894281`。否则 MAVROS 可能切到 `OFFBOARD`，但 PX4 因没有 GCS 拒绝解锁，飞机不会动。

### 3. MAVROS、巡检任务、RViz

```bash
source /opt/ros/humble/setup.bash
source ~/drone-inspection/install/setup.bash
ros2 launch park_inspection inspection.launch.py \
  fcu_url:="udp://:14540@127.0.0.1:14557"
```

终端 2 选的是带雷达的机型时，必须加上 `enable_lidar:=true`。只开这个开关、飞机仍是 `gz_x500_depth` 时，建图会一直等雷达，RViz 里也没有点云。

节点先等飞控连接，在尚未解锁时预先算完整条巡检路线，然后再预发布 Offboard setpoint、请求 Offboard 和解锁。请先确认 Gazebo 和 PX4 都正常，再启动这一步。

飞完后按终端 3、2、1 的顺序 `Ctrl+C`。

## Launch 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `fcu_url` | `udp://:14540@127.0.0.1:14557` | MAVROS 连 PX4 SITL |
| `gcs_url` | 空 | 可选地面站 |
| `enable_mapping` | `true` | 桥接深度相机并做地图对比 |
| `enable_lidar` | `false` | 再桥接 16 线雷达，需要 `x500_depth_lidar` |
| `use_rviz` | `true` | 打开 RViz2 |
| `enable_exploration` | `false` | 发布前沿游览，不改巡检航线 |
| `enable_online_replanning` | `false` | 实验性融合地图重规划 |

常用组合：

```bash
# 不开图形界面
ros2 launch park_inspection inspection.launch.py \
  fcu_url:="udp://:14540@127.0.0.1:14557" use_rviz:=false

# 只按已知地图飞，不建图
ros2 launch park_inspection inspection.launch.py \
  fcu_url:="udp://:14540@127.0.0.1:14557" \
  enable_mapping:=false use_rviz:=false

# 深度 + 雷达 + 前沿可视化
ros2 launch park_inspection inspection.launch.py \
  fcu_url:="udp://:14540@127.0.0.1:14557" \
  enable_lidar:=true enable_exploration:=true
```

不要把 `enable_online_replanning:=true` 和 `enable_mapping:=false` 一起用，任务会因为没有新鲜地图而不解锁。

## 坐标系

规划地图、Gazebo 世界和 MAVROS 本地位置都是 **ENU**（东、北、上）。PX4 内部是 NED，MAVROS 负责转换。RViz 的 Fixed Frame 是 `map`。

深度点云在传感器系：+X 前、+Y 左、+Z 上（`cloud_convention: x_forward`）。雷达光学中心在机顶，相对 `base_link` 为 `(0, 0, 0.09)` m。

## 园区场景

围墙范围大约 `x ∈ [-95, 135]`、`y ∈ [-95, 95]`。规划边界是 `(-100, -100, -1)` 到 `(140, 100, 40)`。

**内园** 160 m × 120 m：南北向街道 `x = -20, 20, 60`，东西向 `y = -20, 20`，外加环路。11 个街区、30 栋楼，布局有四塔组团、双板楼、塔楼加裙房、大体量裙楼。沥青路有中心虚线和斑马线，人行道路缘高 8 cm。

**外圈**：西南停车场（12 辆静态车、画线车位），北侧足球场和球门，体育馆 / 食堂 / 仓库 / 访客中心，西侧树林公园。

**围栏与大门**：铁艺栏杆、混凝土墙基、砖柱。南大门在 `x = 20`，有门柱、门楣、门卫亭、道闸和移门，门外接公路。

**立面**：低于 18 m 的楼是凹窗（窗框、中梃、横档）；更高的塔楼是竖梃幕墙加层间横带。主入口一侧底层后退成柱廊玻璃门厅，有双开门和雨棚。楼顶有女儿墙和设备箱。

**路侧**：约 65 盏 8 m 挑臂路灯、细杆标识牌、矮灌木绿篱、273 棵乔木（三种树形，树干 + 分枝 + 低多边形树冠）。

外观网格的顶点都落在对应规划盒里，传感器看到的东西不会超出规划要避开的范围。

默认巡检点（ENU 米）：

| 顺序 | 位置 | 含义 |
| --- | --- | --- |
| 1 | `(40, 0, 8)` | 中心组团楼缝 |
| 2 | `(80, 0, 10)` | 东侧板楼峡谷 |
| 3 | `(33, -37, 12)` | 东南塔楼立面旁 |
| 4 | `(-40, -40, 10)` | 西南四塔巷道 |
| 5 | `(0, 40, 7)` | 北侧宿舍组团 |

返航高度 6 m。整条航线约 354 m。

改布局只编辑 `park_inspection/campus.py`，立面/树形/贴图在 `tools/generate_campus.py` 和 `tools/meshgen.py`，然后：

```bash
cd ~/drone-inspection/src/park_inspection
python3 tools/generate_campus.py
cd ~/drone-inspection
colcon build --symlink-install --packages-select park_inspection
```

生成器会检查网格顶点是否超出规划盒。当前输出大约 200 个网格、34 万三角面、40 MB。

## 规划与飞行

`GridPlanner3D` 在 1 m 栅格上做 26 连通 A\*，障碍四周留 `obstacle_clearance`（默认 2.5 m）。用 4 m 的平面桶和逐格缓存加速碰撞查询；斜步和视线平滑对膨胀盒做精确线段相交，避免只抽中点时擦进裕度。

任务节点在解锁前算好每一段。飞行中 20 Hz 发布当前位置的下一个路径点。到达判据是 `acceptance_radius`（默认 1.0 m）。任务指标里的 `minimum_safe_clearance`（默认 1.5 m）只用于统计，不改变规划。

## 建图与地图对比

占据图是固定边界上的均匀栅格（分辨率 1 m），不是完整 OctoMap 八叉树，也没有滚动局部地图。定位用 PX4 EKF + MAVROS 本地位置。

- 深度：`max_range` 15 m（相机远裁剪约 19.1 m），`cloud_stride` 16，每帧最多 2500 点，最短周期 0.5 s。
- 雷达：传感器最远 100 m，建图默认只用 `lidar_max_range` 40 m；Gazebo 里 `visualize=false`、5 Hz，避免 GUI 把桌面 GPU 卡死。
- `ground_z` 0.4 m 以下的回波只清空空间，不记障碍。
- 未知体素保持自由。

地图对比在后台线程里用占据快照规划三条航线，最多每 10 秒一次，整个园区大约 11 秒，不占用 Offboard 周期。仅传感器路线若穿过尚未看到的已知建筑，会计入 `online_path_prior_intersections`。

## 实验功能

### 前沿探索（`enable_exploration:=true`）

在在线栅格上维护「已知自由、紧邻未知」的前沿簇，大簇沿最长轴切开，为每个簇采样观察点，近邻 + 2-opt 排开放游览，再用现有 A\* 连到第一个观察点。RViz 里橙色小块是前沿，黄线是到下一观察点的路径。

没有移植 FUEL 的 B 样条最短时间轨迹，也没有 LKH。这条路径**不会**替换 Offboard 指令。

### 在线局部重规划（`enable_online_replanning:=true`）

任务节点等建图状态变成 `mapping_active`、占据快照新鲜后才解锁。当前巡检段和在线障碍冲突时，后台线程重规划到当前目标，控制回调继续发当前位置 setpoint。超时或没有安全路线则任务失败并请求 `AUTO.LOITER`。

已知障碍仍是硬约束，未知体素仍按可通行处理。Python 线程不是硬实时，超时也不能杀掉还在跑的规划线程。

**这不是经验证的避碰系统。** 传感器漏检、误报、位姿误差或障碍进入制动距离内都可能失败。`AUTO.LOITER` 不保证一定能安全悬停。只在可控 SITL 里打开。

## 话题与输出文件

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `/inspection/planned_path` | `nav_msgs/Path` | 当前规划路径 |
| `/inspection/map_markers` | `visualization_msgs/MarkerArray` | 已知障碍（橙色）和巡检点 |
| `/inspection/online_map` | `MarkerArray` | 在线占据（青色） |
| `/inspection/lidar_points` | `sensor_msgs/PointCloud2` | 变到地图系的雷达点 |
| `/inspection/map_comparison` | `std_msgs/String` | 三条路线对比 JSON |
| `/inspection/online_occupancy` | `std_msgs/String` | 占据快照：边界、分辨率、`generation`、行优先 `blocked_indices` |
| `/inspection/mapper_status` | `std_msgs/String` | 建图链路诊断 |
| `/inspection/metrics` | `std_msgs/String` | 任务指标 |
| `/inspection/exploration_frontiers` | `MarkerArray` | 前沿簇（探索打开时） |
| `/inspection/exploration_path` | `nav_msgs/Path` | 到下一观察点（探索打开时） |

原始传感器：深度 `/depth_camera/points`，雷达 `/lidar/points`。

```bash
ros2 topic echo /inspection/metrics
ros2 topic echo /inspection/mapper_status
ros2 topic echo /inspection/map_comparison
```

任务结束后指标默认写到 `~/park_inspection_metrics.json`（成功状态、覆盖率、航程、耗时、最小障碍间距、安全裕度告警、setpoint 间隔统计）。建图节点在收到位姿后至少写一次 `~/park_inspection_map_comparison.json`，之后地图有更新再刷新。

`/inspection/mapper_status` 每秒一条。启用的传感器都有新数据且至少各处理到一个有效回波时，`status` 为 `mapping_active`。其他值：`waiting_for_pose`、`waiting_for_sensor_data`、`partial_sensor_data`、`sensor_data_stale`、`waiting_for_sensor_integration`。这是链路诊断，不代表定位或地图质量达标。

RViz：Fixed Frame `map`。橙色已知障碍，青色在线体素，蓝线规划路径，彩色点是雷达点云。探索打开时还有橙色前沿和黄线。

## 配置

两份 YAML 都挂在 `/**` 下，任务节点和建图节点共用。

**`config/park_mvp.yaml`**

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `grid_resolution` | 1.0 m | 规划/占据栅格 |
| `obstacle_clearance` | 2.5 m | 规划绕障裕度 |
| `minimum_safe_clearance` | 1.5 m | 轨迹评估阈值 |
| `acceptance_radius` | 1.0 m | 到达判据 |
| `sensor_position_body` | OakD-Lite 安装位 | 深度相机相对 `base_link` |
| `lidar_position_body` | `[0, 0, 0.09]` | 雷达光学中心 |
| `max_range` / `lidar_max_range` | 15 / 40 m | 建图使用距离 |
| `ground_z` | 0.4 m | 地面回波阈值 |
| `comparison_period` | 10 s | 地图对比周期 |
| `enable_lidar` / `enable_exploration` | false | 也可被 launch 覆盖 |

**`config/campus_obstacles.yaml`**（生成）

障碍每 6 个数一组：`[中心 x, 中心 y, 中心 z, 尺寸 x, 尺寸 y, 尺寸 z]`。巡检点每 3 个数一组。必须写成浮点（`40.0` 而不是 `40`），否则 ROS 会当成整数数组。地图下界在地面下留定位容差；巡检航点仍应在地面以上。

## 测试

```bash
source /opt/ros/humble/setup.bash
cd ~/drone-inspection/src/park_inspection
python3 -m pytest -q test
```

| 文件 | 覆盖 |
| --- | --- |
| `test_campus.py` | 生成文件与布局一致；网格/贴图引用可解析；每个 link 有碰撞和惯性；碰撞落在规划盒内；楼间距 ≥ 8 m；树/灯/牌/篱/车不碰楼；默认航线可规划 |
| `test_planning.py` | A\*、平滑、膨胀 |
| `test_occupancy.py` | 射线更新、膨胀、路线对比 |
| `test_exploration.py` | 前沿簇与观察点 |
| `test_mission_node.py` | 任务状态机（不接飞控） |

## 常见问题

**电脑卡死、风扇狂转**  
雷达在 Gazebo GUI 里画射线会把桌面 GPU 打满。当前模型已是 `visualize=false`、5 Hz。点云看 RViz 的 `/inspection/lidar_points`。GUI 仍然卡时，用无界面 Gazebo，或先关掉雷达。

**`libEGL warning: egl: failed to create dri2 screen`**  
无害。Mesa EGL 先失败，随后会用 NVIDIA EGL。

**飞机解锁后不动**  
在 `pxh>` 确认 `NAV_DLL_ACT` 是 `0`。没设的话 PX4 会因没有 GCS 拒绝解锁。

**飞机生成时陷进地里或弹飞**  
漏了 `PX4_GZ_MODEL_POSE="0,0,0.35"`。起飞坪有 9 cm 高的碰撞。

**建图一直等雷达 / RViz 没有点云**  
`enable_lidar:=true` 必须搭配 `x500_depth_lidar`。对比文件不出现时，先看 `/inspection/mapper_status`：`pose_received` 应为 true，已启用传感器的 `frames_received` / `frames_integrated` 应递增。

**世界里网格是粉色或缺失**  
`park.sdf` 被单独拷走了，相对 `meshes/` 路径断了。请打开仓库或 install 前缀里的原文件。

**`InvalidParameterTypeException: INTEGER_ARRAY`**  
巡检点或障碍写成了整数。重新跑生成器，YAML 必须是浮点。

## 后续方向

1. 在 GNSS 退化、定位误差和动态障碍下重复飞行，并把扰动写进指标。
2. 做完在线重规划的 PX4/Gazebo 闭环验证：碰撞、超时、重规划时长、失败后的实际模式。
3. 在未知空间有安全策略之前，不要改成只靠传感器地图飞。
4. 设施视角/朝向规划和多目标路线。前沿游览已经能算出来并显示，还没有接到 Offboard 上。

定位仍是 PX4 SITL / MAVROS 本地位置，没有在线 SLAM。

## 许可证

MIT。见 [LICENSE](LICENSE)。
