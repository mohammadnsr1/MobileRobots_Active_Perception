# Active Perception Navigation Contribution

This package contains an independent navigation-side contribution for the
`MobileRobots_Active_Perception` workspace. It does not overwrite the existing
`active_perception` package; instead, it adds a separate ROS 2 package focused
on viewpoint-to-viewpoint motion for active perception.

## Included nodes

- `ap_nav_confidence_evaluator`
  - Computes a lightweight confidence score from recent target pose samples.
- `ap_nav_nbv_planner`
  - Samples candidate viewpoints around the target and publishes the selected
    next-best-view pose.
- `ap_nav_orchestrator`
  - Bridges the active perception loop to Nav2 by sending
    `NavigateToPose` goals and monitoring results.
- `ap_nav_safety_monitor`
  - Publishes a simple safe-stop state based on odometry freshness and blocks
    navigation when the robot state is stale.

## Default interfaces

- Input target pose: `/active_perception/target_pose`
- Navigation confidence: `/active_perception/navigation/confidence`
- Navigation NBV required flag: `/active_perception/navigation/nbv_required`
- Navigation NBV goal: `/active_perception/navigation/nbv_goal`
- Safety state: `/active_perception/navigation/safe_to_navigate`
- Safety status text: `/active_perception/navigation/safety_status`
- Nav2 action: `navigate_to_pose`

## Design intent

The package matches the navigation role described in the project report:

- perception estimates the object pose,
- navigation evaluates whether another view is needed,
- the NBV planner selects a better viewpoint,
- the orchestrator sends that viewpoint to Nav2,
- the safety monitor can force a safe stop if odometry becomes stale.
