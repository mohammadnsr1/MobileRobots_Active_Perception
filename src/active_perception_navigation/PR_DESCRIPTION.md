## What changed

This PR adds a new standalone ROS 2 package, `active_perception_navigation`, to
contribute the navigation side of the active perception project without
modifying the existing `active_perception` package.

The package includes:

- a topic-based confidence evaluator for repeated target pose observations
- a next-best-view planner that samples viewpoints around the target
- a Nav2-based orchestrator that sends `NavigateToPose` goals
- a lightweight safety monitor that blocks navigation when odometry becomes stale
- a launch file to bring the navigation contribution up as a small pipeline
- package-level documentation describing how this contribution fits into the project

## Why it changed

The navigation role in this project is to bridge perception uncertainty and
physical robot motion on TurtleBot4. This contribution matches that role by
keeping Nav2 as the motion planning framework while adding custom logic for:

- viewpoint-to-viewpoint movement
- next-best-view selection
- action-based goal dispatch
- basic safe-stop behavior

Keeping this work in a separate package preserves the original repository code
and makes the contribution easy to review independently.

## Impact

- Adds an isolated navigation contribution under `src/active_perception_navigation`
- Avoids overwriting upstream active perception logic
- Gives the workspace a reusable launchable navigation pipeline for active perception experiments
- Aligns the implemented code more closely with the navigation report and role definition

## Validation

- `python3 -m compileall src/active_perception_navigation`
