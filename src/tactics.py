"""Pure tactical decisions for the match strategy.

Everything in this module is a pure function of the current frame snapshot: it
imports only the data contract and geometry helpers, keeps no cross-frame
state, and never touches a robot. ``main.py`` wires the results to ``Player``
actions, and the decisions stay unit-testable without the simulation.

The two rules that shape most of the code here:

* A goal scored by the kicking team straight from a restart spot does not count.
  The ball must be touched by two different robots, or touched more than 0.5 m
  from the spot. So restarts nudge the ball clear instead of shooting.
* Our own throw-in, goal kick and corner must not send the first touch toward
  our own goal, which would be an own goal.
"""

from __future__ import annotations

import math

from .framework.types import Context
from .param import *
from .utils.geom import clamp, clamp_inside_field, dist, own_goal


def kickoff_point(context: Context) -> tuple[float, float]:
    """Return the ball spot for a kick-off restart: the centre mark."""
    return (0.0, 0.0)


def ball_left_restart(
    context: Context, origin: tuple[float, float], clear_m: float,
) -> bool:
    """Whether the ball has left a restart spot far enough to play normally.

    ``clear_m`` must stay above the rule's 0.5 m threshold so the second touch
    is unambiguously outside it.
    """
    ball = context.ball
    if ball is None:
        return False
    return dist(ball.x, ball.y, origin[0], origin[1]) >= clear_m


def _openness(context: Context, x: float, y: float) -> float:
    """Return the distance from the nearest visible opponent to (x, y).

    Larger means more space. With no opponent visible the point is treated as
    wide open with a finite value, so the lateral-offset penalty can still
    break ties toward a straight shot.
    """
    best = math.inf
    for robot in context.opponents.values():
        pose = robot.pose
        if pose is None:
            continue
        best = min(best, dist(pose.x, pose.y, x, y))
    return SHOT_OPEN_GOAL_CLEARANCE_M if best == math.inf else best


def _free_side(
    context: Context, x: float, y: float, forward_m: float, lateral_m: float,
) -> float:
    """Return +1 or -1 for the more open side of a point ``forward_m`` ahead."""
    up = _openness(context, x + forward_m, y + lateral_m)
    down = _openness(context, x + forward_m, y - lateral_m)
    return 1.0 if up >= down else -1.0


def kickoff_push_target(context: Context) -> tuple[float, float]:
    """Pick where the kick-off taker nudges the ball before playing on.

    A straight shot from the centre mark cannot score, so the first touch only
    has to move the ball clear of the spot and off the goal-mouth line. Aiming
    to the open side also takes the ball away from the opponent most likely to
    intercept.
    """
    ball = context.ball
    if ball is None:
        return kickoff_point(context)
    side = _free_side(
        context, ball.x, ball.y,
        KICKOFF_PUSH_FORWARD_M, KICKOFF_PUSH_LATERAL_M,
    )
    return clamp_inside_field(
        context,
        ball.x + KICKOFF_PUSH_FORWARD_M,
        ball.y + side * KICKOFF_PUSH_LATERAL_M,
        margin=RESTART_FIELD_MARGIN_M,
    )


def set_play_push_target(context: Context) -> tuple[float, float]:
    """Pick where our set-play taker nudges the ball before playing on.

    The target always lies toward the opponent's half, so the first touch can
    never travel toward our own goal, and it is pulled toward the centre line
    so a corner is not dribbled into the side netting.
    """
    ball = context.ball
    if ball is None:
        return (0.0, 0.0)
    return clamp_inside_field(
        context,
        ball.x + SET_PLAY_PUSH_FORWARD_M,
        ball.y * SET_PLAY_PUSH_CENTRE_GAIN,
        margin=RESTART_FIELD_MARGIN_M,
    )


def choose_shot_target(context: Context) -> tuple[float, float]:
    """Aim at the emptiest part of the opponent's goal mouth.

    Candidates are spread across the mouth, keeping a margin from each post.
    Each is scored by the distance to the nearest opponent, with a small
    penalty for shooting further away from the ball's line, which keeps the
    choice close to straight-on when the mouth is empty.
    """
    field = context.field
    goal_x = field.length / 2.0
    ball = context.ball
    if ball is None:
        return (goal_x, 0.0)

    half = field.goal_width / 2.0 - SHOT_AIM_POST_MARGIN_M
    if half <= 0.0:
        return (goal_x, 0.0)

    steps = max(SHOT_AIM_CANDIDATES, 1)
    best_y, best_score = 0.0, -math.inf
    for index in range(steps):
        frac = 0.0 if steps == 1 else -1.0 + 2.0 * index / (steps - 1)
        target_y = frac * half
        offset_penalty = (
            SHOT_AIM_OFFSET_PENALTY
            * abs(target_y - ball.y)
            / max(field.goal_width, 1e-6)
        )
        score = _openness(context, goal_x, target_y) - offset_penalty
        if score > best_score:
            best_y, best_score = target_y, score
    return (goal_x, best_y)


def guard_target(context: Context) -> tuple[float, float]:
    """Return the goalkeeper position: on the goal line, tracking the ball.

    Standing on the line inside the goal mouth beats sitting at the goal-area
    centre, because it covers a shot from any angle and stays legal.
    """
    field = context.field
    goal_x = -field.length / 2.0 + GUARD_LINE_OFFSET_M
    ball = context.ball
    if ball is None:
        return (goal_x, 0.0)
    limit = max(field.goal_width / 2.0 - GUARD_POST_MARGIN_M, 0.0)
    return (goal_x, clamp(ball.y * GUARD_TRACK_GAIN, -limit, limit))


def second_attacker_target(context: Context) -> tuple[float, float]:
    """Hold a receiving position ahead of the ball, on the open side."""
    ball = context.ball
    if ball is None:
        return (0.0, 0.0)
    side = _free_side(
        context, ball.x, ball.y,
        SECOND_ATTACK_FORWARD_M, SECOND_ATTACK_LATERAL_M,
    )
    return clamp_inside_field(
        context,
        ball.x + SECOND_ATTACK_FORWARD_M,
        ball.y + side * SECOND_ATTACK_LATERAL_M,
        margin=RESTART_FIELD_MARGIN_M,
    )


def ready_slots(
    context: Context, player_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Assign READY positions by player id rather than by list position.

    Indexing the player list mis-assigns slots as soon as one player is
    penalised and the list shortens, which puts the wrong robot on the centre
    circle and can draw an illegal-positioning penalty. Player ids run 1..N, so
    each id owns a fixed slot and a penalised teammate leaves the others where
    they are instead of shifting them.

    The first slot stays outside the centre circle for both kick-off sides:
    standing on the line is a coin flip with the referee's tolerance, and the
    taking robot can walk in during PLAYING.
    """
    field = context.field
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    circle_slot = (-field.circle_radius - KICKOFF_CIRCLE_STANDOFF_M, 0.0)
    goal_area_slot = (-field.length / 2.0 + field.goal_area_length, 0.0)
    if our_kickoff:
        slots = [
            circle_slot,
            goal_area_slot,
            (-0.5, field.circle_radius + 2.0),
        ]
    else:
        slots = [
            circle_slot,
            goal_area_slot,
            (-field.length / 2.0 + field.penalty_area_length, 0.0),
        ]

    assigned: dict[int, tuple[float, float]] = {}
    taken: set[int] = set()
    for player_id in sorted(player_ids):
        index = (player_id - 1) % len(slots)
        # Keep the player's own slot, falling back to the next free one only if
        # ids outside 1..N collide.
        for offset in range(len(slots)):
            candidate = (index + offset) % len(slots)
            if candidate not in taken:
                taken.add(candidate)
                assigned[player_id] = slots[candidate]
                break
    return assigned


def assign_normal_roles(
    context: Context,
    candidates: list[tuple[int, tuple[float, float]]],
    locked_attacker: int | None = None,
) -> dict[int, str]:
    """Return the role name for each player id during normal play.

    The nearest player attacks, the next one joins the attack once there are
    three players, and the remaining one guards. Forcing the ball carrier to
    beat a single defender is the framework's baseline; a second attacker turns
    two of the three robots into a threat, while the guard keeps the goal
    covered.

    ``locked_attacker`` keeps the current ball carrier in the role until
    another player is clearly closer, so distance sorting cannot swap roles
    back and forth and interrupt a chase mid-frame.
    """
    if not candidates:
        return {}

    positions = dict(candidates)
    ids = [player_id for player_id, _ in candidates]
    ball = context.ball
    if ball is None:
        return {player_id: "guard" for player_id in ids}

    gx, gy = own_goal(context)

    def ball_distance(player_id: int) -> float:
        x, y = positions[player_id]
        return dist(x, y, ball.x, ball.y)

    def goal_distance(player_id: int) -> float:
        x, y = positions[player_id]
        return dist(x, y, gx, gy)

    attacker = min(ids, key=ball_distance)
    if (
        locked_attacker is not None
        and locked_attacker in positions
        and ball_distance(locked_attacker) <= ball_distance(attacker) + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        attacker = locked_attacker

    roles = {attacker: "attack"}
    rest = [player_id for player_id in ids if player_id != attacker]

    if len(rest) >= 2:
        helper = min(rest, key=ball_distance)
        roles[helper] = "attack2"
        rest = [player_id for player_id in rest if player_id != helper]

    if rest:
        guard = min(rest, key=goal_distance)
        roles[guard] = "guard"
        rest = [player_id for player_id in rest if player_id != guard]

    for player_id in rest:
        roles[player_id] = "support"
    return roles
