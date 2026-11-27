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


def kickoff_push_target(context: Context) -> tuple[float, float]:
    """Pick where the kick-off taker puts the ball: to its own left.

    A shot from the centre mark cannot count, so the first touch is a pass.
    The taker faces the opponent goal, which puts its left hand on +y, and the
    second attacker waits on that side: passing to a fixed side is what makes
    the pair rehearsed, while picking the emptier side frame to frame means
    neither player knows where the ball is going.
    """
    ball = context.ball
    if ball is None:
        return kickoff_point(context)
    return clamp_inside_field(
        context,
        ball.x + KICKOFF_PUSH_FORWARD_M,
        ball.y + KICKOFF_PUSH_LATERAL_M,
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


def _lane_blocked(
    context: Context,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> bool:
    """Whether an opponent stands within ``radius`` of the start-to-end segment.

    This is the pass/shot lane test: a body on the line between the ball and
    the target is what turns a good idea into a turnover.
    """
    ax, ay = start
    bx, by = end
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    for robot in context.opponents.values():
        pose = robot.pose
        if pose is None:
            continue
        if length2 < 1e-9:
            t = 0.0
        else:
            t = clamp(((pose.x - ax) * vx + (pose.y - ay) * vy) / length2, 0.0, 1.0)
        px, py = ax + vx * t, ay + vy * t
        if math.hypot(pose.x - px, pose.y - py) <= radius:
            return True
    return False


def choose_pass_target(
    context: Context, receiver_pos: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Return the receiver when a pass beats shooting, or None to shoot.

    Passing is only worth it when all three hold: the straight shot lane is
    blocked, the lane to the receiver is clean, and the receiver is further
    forward than the ball. Otherwise the shot stays the better option and a
    pass would hand possession to the nearest defender.
    """
    ball = context.ball
    if ball is None or receiver_pos is None:
        return None

    field = context.field
    ball_pos = (ball.x, ball.y)
    goal = (field.length / 2.0, 0.0)

    if not _lane_blocked(context, ball_pos, goal, PASS_SHOT_LANE_BLOCK_M):
        return None
    if _lane_blocked(context, ball_pos, receiver_pos, PASS_LANE_BLOCK_M):
        return None
    if receiver_pos[0] <= ball.x + PASS_MIN_ADVANCE_M:
        return None
    return receiver_pos


def guard_target(context: Context) -> tuple[float, float]:
    """Return the goalkeeper position, on the goal-centre-to-ball line.

    Mirroring the ball's own y puts the keeper at the near post whenever the
    ball is wide, while the shot travels toward the goal centre — so the ball
    passes a keeper that is standing in the right place for the wrong line and
    arrives too late. Placing the keeper where the goal-centre-to-ball line
    crosses the keeper's line keeps the body on the shot's actual path.

    When the ball is almost on the goal line the intersection is ill-defined,
    so the keeper simply mirrors it and covers the short side.
    """
    field = context.field
    goal_line = -field.length / 2.0
    goal_x = goal_line + GUARD_LINE_OFFSET_M
    ball = context.ball
    if ball is None:
        return (goal_x, 0.0)

    limit = max(field.goal_width / 2.0 - GUARD_POST_MARGIN_M, 0.0)
    depth = ball.x - goal_line
    if depth <= GUARD_NEAR_BALL_M:
        y = ball.y
    else:
        y = ball.y * GUARD_LINE_OFFSET_M / depth
    return (goal_x, clamp(y, -limit, limit))


def second_attacker_target(
    context: Context,
    prev_target: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return the receiving slot for the second attacker.

    Evaluates flanks relative to the ball plus central and weak-side lanes so
    when the ball is pinned on one touchline the second attacker runs into the
    open middle/far side instead of crowding the touchline. Optional
    ``prev_target`` adds hysteresis so noisy opponent movement cannot flip the
    target every frame and leave the robot turning in place.
    """
    ball = context.ball
    if ball is None:
        return (0.0, 0.0)

    ball_pos = (ball.x, ball.y)
    tx = ball.x + SECOND_ATTACK_FORWARD_M
    weak_y = -SECOND_ATTACK_LATERAL_M if ball.y >= 0.0 else SECOND_ATTACK_LATERAL_M
    candidate_ys = (
        ball.y + SECOND_ATTACK_LATERAL_M,
        ball.y - SECOND_ATTACK_LATERAL_M,
        ball.y,
        0.0,
        weak_y,
    )

    best = clamp_inside_field(
        context, tx, ball.y + SECOND_ATTACK_LATERAL_M, margin=RESTART_FIELD_MARGIN_M,
    )
    best_score = -math.inf
    for cy in candidate_ys:
        spot = clamp_inside_field(
            context, tx, cy, margin=RESTART_FIELD_MARGIN_M,
        )
        score = _openness(context, spot[0], spot[1])
        if _lane_blocked(context, ball_pos, spot, RECEIVE_LANE_CLEAR_M):
            score -= RECEIVE_SCREEN_PENALTY
        if prev_target is not None:
            d_prev = dist(spot[0], spot[1], prev_target[0], prev_target[1])
            score += SECOND_ATTACK_HYSTERESIS * max(0.0, 1.0 - d_prev / 2.5)
        if score > best_score:
            best, best_score = spot, score
    return best


def measure_ball_velocity(
    prev: tuple[float, float, float] | None,
    ball,
    now: float,
) -> tuple[float, float]:
    """Estimate the ball's velocity from the previous frame's snapshot.

    ``prev`` is ``(x, y, now)``. Restarts teleport the ball, which would read
    as an enormous velocity and send the receiver sprinting at nothing, so a
    speed above ``BALL_MAX_TRACKED_SPEED`` is reported as zero instead of
    trusted. A missing or repeated snapshot also reports zero.
    """
    if prev is None or ball is None:
        return (0.0, 0.0)
    px, py, prev_now = prev
    dt = now - prev_now
    if dt <= 1e-6:
        return (0.0, 0.0)
    vx = (ball.x - px) / dt
    vy = (ball.y - py) / dt
    if math.hypot(vx, vy) > BALL_MAX_TRACKED_SPEED:
        return (0.0, 0.0)
    return (vx, vy)


def blend_velocity(
    previous: tuple[float, float], measured: tuple[float, float],
) -> tuple[float, float]:
    """Smooth a velocity estimate so one noisy frame cannot swing the target."""
    alpha = BALL_VELOCITY_SMOOTHING
    return (
        previous[0] * (1.0 - alpha) + measured[0] * alpha,
        previous[1] * (1.0 - alpha) + measured[1] * alpha,
    )


def receiving_target(
    context: Context,
    pose: tuple[float, float],
    ball_velocity: tuple[float, float] = (0.0, 0.0),
    prev_target: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return where the receiver should run: the ball's future position.

    Only leads a ball that is genuinely travelling toward an ahead/level
    receiver; minor backward bounces during a midfield duel keep the steady
    open receiving slot so the second attacker pushes upfield instead of
    stalling on the halfway line.
    """
    ball = context.ball
    if ball is None:
        return (0.0, 0.0)

    vx, vy = ball_velocity
    speed = math.hypot(vx, vy)
    if speed < RECEIVE_LEAD_MIN_SPEED:
        return second_attacker_target(context, prev_target=prev_target)

    dx, dy = pose[0] - ball.x, pose[1] - ball.y
    d_recv = math.hypot(dx, dy)
    if d_recv < 1e-6 or pose[0] < ball.x - 0.5:
        return second_attacker_target(context, prev_target=prev_target)

    cos_align = (dx * vx + dy * vy) / (speed * d_recv)
    if cos_align < RECEIVE_LEAD_MIN_COS:
        return second_attacker_target(context, prev_target=prev_target)

    return clamp_inside_field(
        context,
        ball.x + vx * RECEIVE_LEAD_S,
        ball.y + vy * RECEIVE_LEAD_S,
        margin=RESTART_FIELD_MARGIN_M,
    )


def score_difference(context: Context) -> int:
    """Return our score minus the opponent's, or 0 when the score is unknown."""
    game = context.game
    if game is None:
        return 0
    ours = game.get_team_state(context.team_id)
    if ours is None:
        return 0
    theirs = 0
    for team in game.teams:
        if team.team_number != context.team_id:
            theirs = team.score
            break
    return ours.score - theirs


def should_stall_restart(context: Context) -> bool:
    """Whether we should run down our own restart clock instead of kicking.

    When leading by at least ``STALL_RESTART_LEAD_MARGIN`` (e.g. 2:0), waiting
    for the official restart countdown (10 s on kick-off, 45 s on set plays)
    to expire and release ball_free burns clock legally without risking a
    turnover. All robots must wait > 3.0 m from the ball so M-05 never fires.
    """
    return score_difference(context) >= STALL_RESTART_LEAD_MARGIN


def stall_hold_slots(
    context: Context,
    count: int,
    *,
    own_half_only: bool = False,
) -> list[tuple[float, float]]:
    """Return ``count`` goal-side holding spots strictly > 3.0 m from the ball.

    Used while stalling our own set plays or kick-offs when leading >= 2 goals:
    * Keeps every robot away from the ball so nobody touches it early.
    * Guarantees ``dist(spot, ball) >= STALL_SAFE_DIST_M`` (> 3.0 m) so waiting
      up to 45 s cannot trigger an M-05 no-activity penalty.
    * Biases spots toward our own goal (goal-side defense) and, when
      ``own_half_only`` is True (kick-offs), keeps every spot strictly in our
      own half and outside the centre circle (rules P-02 / P-03).
    """
    if count <= 0:
        return []
    field = context.field
    half_l = field.length / 2.0
    half_w = field.width / 2.0
    margin = RESTART_FIELD_MARGIN_M
    ball = context.ball
    bx, by = (ball.x, ball.y) if ball is not None else (0.0, 0.0)
    gx, gy = own_goal(context)

    if own_half_only:
        # Staggered defensive slots in our own half, > 3.25 m from (0, 0) and
        # well outside the 1.5 m centre circle.
        preset = [
            clamp_inside_field(context, -2.6, 2.2, margin=margin),
            clamp_inside_field(context, -3.8, -1.9, margin=margin),
            clamp_inside_field(context, -half_l + field.goal_area_length, 0.0, margin=margin),
        ]
        return preset[:count]

    # Sample angles around the ball at STALL_HOLD_DIST_M, preferring angles
    # pointing toward our own goal so our players screen the goal while stalling.
    goal_angle = math.atan2(gy - by, gx - bx)
    candidates: list[tuple[float, float, float]] = []
    steps = 24
    for i in range(steps):
        ang = goal_angle + (2.0 * math.pi * i) / steps
        cx = bx + math.cos(ang) * STALL_HOLD_DIST_M
        cy = by + math.sin(ang) * STALL_HOLD_DIST_M
        if not (-half_l + margin <= cx <= half_l - margin):
            continue
        if not (-half_w + margin <= cy <= half_w - margin):
            continue
        if dist(cx, cy, bx, by) < STALL_SAFE_DIST_M:
            continue
        # Lower score = closer to our goal (better defensive cover).
        goal_dist = dist(cx, cy, gx, gy)
        candidates.append((goal_dist, cx, cy))

    candidates.sort(key=lambda item: item[0])

    # Greedily pick well-separated spots (>= 1.8 m apart) so teammates don't crowd.
    chosen: list[tuple[float, float]] = []
    for _, cx, cy in candidates:
        if all(dist(cx, cy, px, py) >= 1.8 for px, py in chosen):
            chosen.append((cx, cy))
            if len(chosen) >= count:
                break

    # Fallback if tight spacing filtered too many points.
    for _, cx, cy in candidates:
        if len(chosen) >= count:
            break
        if (cx, cy) not in chosen:
            chosen.append((cx, cy))

    while len(chosen) < count:
        # Ultimate fallback: opposite side of midfield well away from the ball.
        fx = -2.0 if bx >= 0.0 else 2.0
        fy = -2.0 if by >= 0.0 else 2.0
        chosen.append(clamp_inside_field(context, fx, fy, margin=margin))
    return chosen


def defender_target(context: Context) -> tuple[float, float]:
    """Return the blocking position between the ball and our goal.

    Standing on the goal-centre-to-ball line is what denies the shooting angle,
    which is the job the goalkeeper cannot do alone. The position must sit in
    the ball's actual path, though: a fixed short standoff parks the defender
    near the goal line, where an attacker walks past it at midfield. So the
    standoff tracks the ball — out to ``DEFENDER_DEPTH_MAX_M`` while the ball is
    far, closing to a body's width once the ball is near enough to challenge.
    """
    field = context.field
    goal_line = -field.length / 2.0
    ball = context.ball
    if ball is None:
        return (goal_line + DEFENDER_DEPTH_MAX_M, 0.0)

    dx, dy = ball.x - goal_line, ball.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return (goal_line + DEFENDER_DEPTH_MAX_M, 0.0)

    ux, uy = dx / distance, dy / distance
    standoff = min(distance - DEFENDER_BALL_GAP_M, DEFENDER_DEPTH_MAX_M)
    standoff = max(standoff, 0.0)
    return clamp_inside_field(
        context,
        goal_line + ux * standoff,
        uy * standoff,
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
        # Staggered-x defensive triangle: three distinct x depths (-2.6, -6.0,
        # -4.0) and opposite flanks (+1.9, 0.0, -1.9) so the two field players
        # stay ~4.05 m apart (no crowding) while forming a clean press + relay
        # shape, all > 3 m from centre (M-05 and P-01/P-02 safe).
        slots = [
            (-OPP_KICKOFF_FRONT_BACK_M, OPP_KICKOFF_FRONT_LATERAL_M),
            goal_area_slot,
            (-OPP_KICKOFF_DEEP_BACK_M, -OPP_KICKOFF_DEEP_LATERAL_M),
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
    chase: bool = False,
) -> dict[int, str]:
    """Return the role name for each player id during normal play.

    The nearest player attacks and the remaining player nearest our goal
    guards. When three robots are available the third one either joins the
    attack or drops in front of the goal, depending on the match situation:
    with the ball in our half, or with a lead to protect, only one robot
    pushes up so the goal is defended by two bodies instead of one. Leaving a
    single goalkeeper is how a lead turns into a shoot-out.

    ``chase`` overrides that shape for the seconds right after our own kick-off.
    A ball just played into the opponent half is a free ball, and dropping the
    second player back as if defending a lead means nobody follows it up — the
    defensive shape applies again as soon as the window closes.

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

    # Defending the lead, or defending our own half, costs one attacker. The
    # post-kick-off chase window overrides both: a loose ball is worth more
    # than the extra body until possession settles.
    defending = not chase and (
        ball.x < 0.0 or score_difference(context) >= DEFEND_LEAD_MARGIN
    )

    roles = {attacker: "attack"}
    rest = [player_id for player_id in ids if player_id != attacker]

    if not defending and len(rest) >= 2:
        helper = min(rest, key=ball_distance)
        roles[helper] = "attack2"
        rest = [player_id for player_id in rest if player_id != helper]

    if rest:
        guard = min(rest, key=goal_distance)
        roles[guard] = "guard"
        rest = [player_id for player_id in rest if player_id != guard]

    for player_id in rest:
        roles[player_id] = "defend"
    return roles
