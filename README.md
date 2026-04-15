# Active Perception for Accurate Object Localization and Navigation
The goal of this project is to develop an autonomous mobile robot system capable of accurately localizing a target object using RGB-D perception and actively improving this estimate through motion. The TurtleBot4 will estimate the target object's ground-plane pose relative to the robot and compute a confidence metric representing the reliability of the estimate.

Using an active perception loop, the system will determine the next-best viewpoint that is expected to reduce pose uncertainty. The robot will autonomously navigate to these viewpoints while avoiding obstacles using the ROS2 Nav2 navigation stack or a reactive controller till a desired confidence threshold is achieved.

# Robot Platform
- **Platform:** TurtleBot 4.
- **Base:** Differential drive.
- **Onboard sensors:** RGB-D camera, LiDAR, IMU.

# High-Level System Architecture
```mermaid
flowchart LR

  subgraph Perception
    RGBD[RGBD Camera]
    LIDAR[LiDAR]
    IMU[IMU]
  end

  subgraph ObjectPerception
    PCP[Point Cloud]
    OPE[Object Pose]
  end

  subgraph RobotLocalization
    VO[Visual Odometry]
    EKF[EKF]
  end
  subgraph Planning
    CONF[Confidence Evaluation]
    NBV[Next Best View]
    NAV2[Nav2 Global Planner]
    RC[Reactive Controller]
  end

  subgraph Actuation
    DDC[Diff Drive Controller]
    MHI[Motor Hardware Interface]
  end

  %% Perception → Object perception
  RGBD --> PCP
  PCP --> OPE

  %% Perception → Localization
  RGBD --> VO
  IMU --> EKF
  VO --> EKF
  

  %% Estimation to planning
  OPE --> CONF
  CONF --> NBV
  EKF --> NBV

  %% Planning to actuation
  NBV --> NAV2
  NAV2 --> RC
  LIDAR --> RC
  RC --> DDC
  DDC --> MHI

  %% Styles
  style Perception fill:#ffe6e6,stroke:#333,stroke-width:2px
  style ObjectPerception fill:#fff2cc,stroke:#333,stroke-width:2px
  style RobotLocalization fill:#fff2cc,stroke:#333,stroke-width:2px
  style Planning fill:#e6e6ff,stroke:#333,stroke-width:2px
  style Actuation fill:#d9f2d9,stroke:#333,stroke-width:2px

```

# Git Infrastructure
- **GitHub Page:** https://seasonedleo.github.io/RAS_Mobile_Robotics_Vision/


