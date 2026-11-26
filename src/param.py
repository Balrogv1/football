"""Central configuration point for strategy tuning.

This module contains tunable strategy, movement, kicking, and obstacle
avoidance parameters. Existing values were moved here from their original
modules; make future parameter adjustments here first.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power to this range. Unsupported values may cause a fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Kick power during normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0


# ======================================================================
# Player movement control
# ======================================================================

ARRIVE_DIST = 0.15             # Target arrival threshold (m)
OMNI_DIST = 1.0                # Use omnidirectional control below this distance
TURN_THRESHOLD = 0.5           # Turn in place above this heading error (rad)
MAX_LINEAR = 2.0               # Maximum commanded forward speed (m/s)
MAX_ANGULAR = 2.0              # Maximum commanded angular speed (rad/s)
LINEAR_GAIN = 1.5              # Translation proportional gain
ANGULAR_GAIN = 2.0             # Rotation proportional gain


# ======================================================================
# Player kicking and shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kicking state below this ball distance
KICK_EXIT_M = 2.5              # Leave kicking state above this ball distance
CHASE_BEHIND_M = 0.35          # Distance behind the ball while chasing


# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # Distance from center at which the ball counts as moved

OPP_SET_WALL_DIST_M = 2.0 # Blocking distance during an opponent kickoff

# Opponent set plays: retreat threshold and field boundary margin (m).
# The rule line is 1.45 m, so the target keeps a real margin rather than
# sitting on the referee's tolerance.
OPP_SET_PLAY_CLEARANCE_M = 1.7
OPP_SET_PLAY_FIELD_MARGIN_M = 0.3
# Hold the goal line only when the ball is this far away: the keeper exemption
# covers distance, not standing still, and 3 m is where the no-activity rule
# starts to apply.
OPP_SET_PLAY_KEEPER_MIN_M = 3.2

# Opponent kick-off: waiting players stay outside 3 m of the ball so a long
# kick-off countdown cannot book them for no activity.
OPP_KICKOFF_HOLD_BACK_M = 3.4
OPP_KICKOFF_HOLD_LATERAL_M = 1.2

# Restarts (kick-off and set plays). A goal from the restart spot only counts
# after a second robot touches the ball or a touch beyond 0.5 m, so the taker
# nudges the ball clear first. Keep both thresholds above 0.5 m with margin.
KICKOFF_CLEAR_M = 0.75                 # Ball considered clear of the centre mark
KICKOFF_PUSH_POWER = 3.0               # First-touch power (soft, stays in play)
KICKOFF_PUSH_FORWARD_M = 2.2           # Push target ahead of the ball
KICKOFF_PUSH_LATERAL_M = 1.6           # Push target to the open side
KICKOFF_CIRCLE_STANDOFF_M = 0.25       # Ready slot offset outside the circle
KICKOFF_HOLD_BACK_M = 2.4              # Non-taker hold spot in our own half
KICKOFF_HOLD_LATERAL_M = 2.6           # ... and its lateral offset (keeps > 3 m)

SET_PLAY_CLEAR_M = 0.7                 # Ball considered clear of the set-play spot
SET_PLAY_PUSH_POWER = 3.0
SET_PLAY_PUSH_FORWARD_M = 2.5          # Always toward the opponent half
SET_PLAY_PUSH_CENTRE_GAIN = 0.3        # Pull the push target toward y = 0
RESTART_FIELD_MARGIN_M = 0.6           # Keep restart targets inside the lines

# Second attacker: joins the chase when close, holds a receiving slot when far.
SECOND_ATTACK_CHASE_M = 3.2            # Commit to the ball inside this range
SECOND_ATTACK_FORWARD_M = 1.5          # Receiving slot ahead of the ball
SECOND_ATTACK_LATERAL_M = 2.0          # Receiving slot to the open side

# Goalkeeper: sit on the goal line and track the ball across the mouth.
GUARD_LINE_OFFSET_M = 0.35             # How far off the goal line to stand
GUARD_TRACK_GAIN = 0.8                 # Fraction of the ball's y to mirror
GUARD_POST_MARGIN_M = 0.15             # Stay this far inside each post


# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)

# Shot placement: avoid the keeper instead of always hitting the goal centre.
SHOT_AIM_CANDIDATES = 7                 # Sample points across the goal mouth
SHOT_AIM_POST_MARGIN_M = 0.35           # Keep the aim off the posts (m)
SHOT_AIM_OFFSET_PENALTY = 0.5           # Cost per metre of lateral deviation
SHOT_OPEN_GOAL_CLEARANCE_M = 20.0       # Clearance assumed when no opponent is seen

# ======================================================================
# Obstacle geometry
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball obstacle radius (m)
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth used for obstacle modeling (m)
POST_RADIUS = 0.18                      # Goal-post radius (m)
NET_RADIUS = 0.20                       # Goal-net sample radius (m)
NET_STEP = 0.35                         # Goal-net sampling interval (m)

START_IGNORE = 0.0                      # Ignore obstacles near the start (m)
TARGET_IGNORE = 0.0                     # Ignore obstacles near the target (m)

# ======================================================================
# Global path planner (A* grid planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Fall back to local VFH when disabled
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid cell size; smaller is finer (m)
GLOBAL_FIELD_MARGIN_M = 0.25            # Extra field margin for boundary paths (m)
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Additional obstacle inflation radius (m)
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # Lookahead distance along the path (m)

# ======================================================================
# Local path planner (VFH direction scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe length (m)
PLAN_CLEARANCE = 0.35                   # Minimum clearance for a candidate (m)
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad)
PLAN_MAX_OFFSET = math.radians(100)     # Maximum offset from target heading (rad)

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
