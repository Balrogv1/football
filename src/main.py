"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- utils/: movement, geometry, and obstacle-avoidance helpers such as
  ``opponent_goal``, ``dist``, and ``angle_to``.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from . import tactics
from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import angle_to, clamp_inside_field, dist, opponent_goal, own_goal


_log = logging.getLogger(__name__)


# ======================================================================
# Phase state machine: match phase classification
# ======================================================================


class Phase(Enum):
    """Top-level match phase controlling normal play, restarts, and stops."""
    NORMAL = "normal"              # Normal contest while PLAYING
    OUR_KICKOFF = "our_kickoff"    # Our kickoff during early SET/PLAYING
    OPP_KICKOFF = "opp_kickoff"    # Opponent kickoff; keep clear
    OUR_SET_PLAY = "our_set_play"  # Our free kick, corner, or goal kick
    OPP_SET_PLAY = "opp_set_play"  # Opponent set play; keep clear
    READY = "ready"                # Move into READY positions
    STOPPED = "stopped"            # Non-kickoff SET, INITIAL, FINISHED, or stopped


def get_phase(context: Context) -> Phase:
    """Determine the current phase from the GameController state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: move to ready positions.
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal play or an active kickoff/set play.
    if state == GameState.PLAYING and not g.stopped:
        # For a set play, kicking_team identifies the team taking it.
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # During the kickoff countdown, kicking_team identifies the taker.
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal play.
        return Phase.NORMAL

    # Remain stationary during SET, INITIAL, FINISHED, or a stoppage.
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """Return the active set-play type, or ``SetPlay.NONE`` when unavailable.

    This reads the GameController's ``set_play`` field without identifying the
    taking team. :func:`get_phase` distinguishes OUR_SET_PLAY from OPP_SET_PLAY;
    this function only identifies the type.

    The seven possible values from framework.types.SetPlay are:
    - ``NONE``: no set play, such as normal play or a kickoff
    - ``DIRECT_FREE_KICK``: a direct free kick that may score immediately
    - ``INDIRECT_FREE_KICK``: another player must touch the ball before a goal
    - ``PENALTY_KICK``: a penalty kick
    - ``THROW_IN``: a kick-in from touch
    - ``GOAL_KICK``: a goal kick
    - ``CORNER_KICK``: a corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.kickoff_receiver = None # Receiver of the kick-off pass
        store.kickoff_released_at = None  # When the ball left the centre spot
        store.normal_attacker = None
        # Restart bookkeeping: a goal from a restart spot only counts once the
        # ball has left it, so the origin has to survive across frames.
        store.set_play_type = None
        store.set_play_origin = None
        store.pass_lock_until = 0.0   # Holds a pass decision across frames

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # Handle readiness and retain only players that can act this frame.
        # ensure_ready asynchronously gets up or switches to walk mode without
        # moving. Penalized players also prepare so they can rejoin immediately.
        # Penalized or unready players are excluded from role assignment to avoid
        # selecting an attacker who cannot move.
        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # May prepare, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Exclude players with unknown positions
                p.stop()
            else:
                active.append(p)

        # Dispatch the team once per phase; _act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw every teammate last, including penalized, unready, and stopped
        # players, so markers remain visible during states such as SET.
        for p in players:
            _draw_teammate_marker(p)


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _player_dist_to_position(p: Player, target: tuple[float, float]) -> float:
    """Return infinity when the player's position is unknown."""
    pose = p.pose
    if pose is None:
        return math.inf
    return dist(pose.x, pose.y, *target)


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to the ball's current position."""
    ball = context.ball
    return (
        _player_dist_to_position(p, (ball.x, ball.y)) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _act_normal(context: Context, players: list[Player], store) -> None:
    """Assign roles from the current snapshot.

    ``tactics.assign_normal_roles`` owns the decision, so the role split stays
    testable off the robot. Three players either field two attackers plus a
    guard, or one attacker plus a guard and a covering defender; two players
    always split into one attacker and one guard.
    """
    if not players:
        return

    candidates = [
        (p.id, (p.pose.x, p.pose.y)) for p in players if p.pose is not None
    ]
    roles = tactics.assign_normal_roles(
        context, candidates, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = next(
        (player_id for player_id, role in roles.items() if role == "attack"),
        None,
    )

    by_id = {p.id: p for p in players}
    positions = dict(candidates)

    # The second attacker is the only intended receiver for a pass.
    receiver_pos = next(
        (
            positions[player_id]
            for player_id, role in roles.items()
            if role == "attack2" and player_id in positions
        ),
        None,
    )

    for player_id, role in roles.items():
        player = by_id.get(player_id)
        if player is None:
            continue
        if role == "attack":
            pass_target = _pass_target(context, receiver_pos, store)
            if pass_target is not None:
                player.action = "attack:pass"
                player.pass_to(pass_target)
            else:
                player.action = "attack"
                player.attack()
        elif role == "attack2":
            player.action = "attack2"
            player.second_attack()
        elif role == "guard":
            player.action = "guard"
            player.guard()
        elif role == "defend":
            player.action = "defend"
            player.defend()
        else:
            player.action = "support"
            player.support()


def _pass_target(
    context: Context,
    receiver_pos: tuple[float, float] | None,
    store,
) -> tuple[float, float] | None:
    """Decide whether the ball carrier should pass, holding the choice briefly.

    The decision flips frame to frame as opponents drift around the lane, and a
    carrier that alternates between shooting and passing never commits to
    either. Once a pass is chosen it is held for ``PASS_LOCK_S``.
    """
    if receiver_pos is None:
        return None
    now = context.now
    if now < getattr(store, "pass_lock_until", 0.0):
        return receiver_pos
    target = tactics.choose_pass_target(context, receiver_pos)
    if target is not None:
        store.pass_lock_until = now + PASS_LOCK_S
    return target


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Pass the ball to the taker's left, let the receiver shoot, then play on.

    A goal from the kick-off spot only counts after a second robot touches the
    ball or a touch more than 0.5 m away from it, so the first touch is a pass
    to the second attacker rather than a shot. For a short window after the
    pass the receiver attacks and the taker drops into support: that is what
    turns the pass into a shot instead of leaving both robots chasing the same
    ball.
    """
    if not players:
        return

    active_ids = {p.id for p in players}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in active_ids:
        # Select a new taker and receiver when entering the kickoff phase.
        taker = _select_closest_attacker(context, players)
        store.kickoff_taker = taker.id
        others = [p for p in players if p is not taker]
        store.kickoff_receiver = (
            _select_closest_attacker(context, others).id if others else None
        )
        store.kickoff_released_at = None

    ball = context.ball
    if ball is None:
        for p in players:
            p.action = "stay"
            p.stop()
        return

    if tactics.ball_left_restart(
        context, tactics.kickoff_point(context), KICKOFF_CLEAR_M,
    ):
        # The ball is clear of the spot, so a goal from here is a legal one.
        if _kickoff_receiver_attacks(context, players, store):
            return
        _act_normal(context, players, store)
        return

    attacker_id = store.kickoff_taker
    attacker = next((p for p in players if p.id == attacker_id), None)
    if attacker is None:
        attacker = _select_closest_attacker(context, players)
        store.kickoff_taker = attacker.id

    target = tactics.kickoff_push_target(context)
    push_dir = angle_to(ball.x, ball.y, target[0], target[1])
    attacker.action = "kickoff:pass"
    attacker.kick(push_dir, KICKOFF_PUSH_POWER)

    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        guard.action = "guard"
        guard.guard()
        rest = [p for p in rest if p is not guard]

    # The receiver waits on the taker's left, just behind the halfway line and
    # outside the centre circle. A non-taker may not cross the halfway line or
    # enter the circle until the ball is played, so this is as far forward as
    # the rules allow.
    for p in rest:
        p.action = "kickoff:receive"
        p.walk_to(
            (-KICKOFF_HOLD_BACK_M, KICKOFF_HOLD_LATERAL_M),
            face=push_dir,
            avoid_ball=True,
            avoid_robots=True,
        )


def _kickoff_receiver_attacks(
    context: Context, players: list[Player], store,
) -> bool:
    """Send the receiver at the ball for a short window after the pass.

    Returns False once the window has passed, so normal play takes over. When
    the receiver is unavailable the caller falls straight through to normal
    play rather than leaving the team standing around.
    """
    if store.kickoff_released_at is None:
        store.kickoff_released_at = context.now
    if context.now - store.kickoff_released_at > KICKOFF_RECEIVE_WINDOW_S:
        return False

    by_id = {p.id: p for p in players}
    receiver = by_id.get(store.kickoff_receiver)
    if receiver is None:
        return False

    for p in players:
        if p is receiver:
            p.action = "kickoff:shoot"
            p.attack()
        elif p.id == store.kickoff_taker:
            p.action = "kickoff:support"
            p.second_attack()
        else:
            p.action = "guard"
            p.guard()
    return True


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """Guard with one player while the others wait outside the centre circle.

    The waiting spots sit more than 3 m from the ball, so a kick-off countdown
    cannot book a stationary robot for no activity.
    """
    if not players:
        return
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    guard.action = "guard"
    guard.guard()

    ball = context.ball
    bx, by = (ball.x, ball.y) if ball is not None else (0.0, 0.0)
    rest = [p for p in players if p is not guard]
    slots = [
        (bx - OPP_KICKOFF_HOLD_BACK_M, by + OPP_KICKOFF_HOLD_LATERAL_M),
        (bx - OPP_KICKOFF_HOLD_BACK_M, by - OPP_KICKOFF_HOLD_LATERAL_M),
    ]
    for p, target in zip(rest, slots):
        p.action = "opp_kickoff:hold"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Push the ball off the restart spot, then play on normally.

    Every set play is subject to the same rule as the kick-off: a goal straight
    from the spot does not count, so the taker's first touch nudges the ball
    clear of it. The push target is always toward the opponent's half, which
    also keeps the first touch away from our own goal.

    The spot is remembered in ``store`` because the ball drifts over several
    frames; the origin resets whenever the set-play type changes.
    """
    if not players:
        return

    set_play = get_set_play_type(context)
    ball = context.ball
    if (
        store.prev_phase != Phase.OUR_SET_PLAY
        or store.set_play_type != set_play
        or store.set_play_origin is None
    ):
        store.set_play_type = set_play
        store.set_play_origin = (ball.x, ball.y) if ball is not None else None

    if ball is None or store.set_play_origin is None:
        for p in players:
            p.action = "set:wait"
            p.stop()
        return

    if tactics.ball_left_restart(
        context, store.set_play_origin, SET_PLAY_CLEAR_M,
    ):
        # The ball is clear of the spot, so a goal from here counts.
        _act_normal(context, players, store)
        return

    taker = _select_closest_attacker(context, players)
    target = tactics.set_play_push_target(context)
    push_dir = angle_to(ball.x, ball.y, target[0], target[1])
    taker.action = "set:push"
    taker.kick(push_dir, SET_PLAY_PUSH_POWER)

    rest = [p for p in players if p is not taker]
    if len(rest) >= 2:
        # A runner follows the push so the second touch is available at once.
        runner = _select_closest_attacker(context, rest)
        runner.action = "set:follow"
        runner.walk_to(target, face=push_dir, avoid_ball=True, avoid_robots=True)
        rest = [p for p in rest if p is not runner]

    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        guard.action = "guard"
        guard.guard()


def _select_opp_set_keeper(
    context: Context, players: list[Player],
) -> Player | None:
    """Pick the player who may hold the goal line during an opponent restart.

    Returns None when the ball is close enough that standing still on the line
    would break either the 1.45 m clearance or the no-activity rule, or when
    only one player is available and the retreat matters more.
    """
    ball = context.ball
    if ball is None or len(players) < 2:
        return None
    gx, gy = own_goal(context)
    candidate = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    home = tactics.guard_target(context)
    if dist(ball.x, ball.y, home[0], home[1]) < OPP_SET_PLAY_KEEPER_MIN_M:
        return None
    return candidate


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Keep clear of the ball, leaving one player on the goal line when safe.

    The referee penalises any defender inside 1.45 m of the restart spot before
    the ball is played, so retreating is the default. A goalkeeper standing on
    the goal line inside the mouth is exempt, and abandoning the goal on every
    opponent restart is how easy goals go in, so one player holds the line when
    the ball is far enough away that neither rule can bite.
    """
    ball = context.ball
    keeper = _select_opp_set_keeper(context, players)
    # Account for walk_to's arrival tolerance when targeting the 1.5 m boundary.
    radius = OPP_SET_PLAY_CLEARANCE_M + ARRIVE_DIST
    half_l = context.field.length / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    for p in players:
        if p is keeper:
            p.action = "guard"
            p.guard()
            continue
        pose = p.pose
        p.action = "opp_set:wait"
        if ball is None or pose is None:
            p.stop()
            continue
        if dist(pose.x, pose.y, ball.x, ball.y) >= OPP_SET_PLAY_CLEARANCE_M:
            p.stop()
            continue

        # The radial exit is shortest. At the field edge, also consider the
        # retreat circle's intersections with the in-field boundary.
        outward = angle_to(ball.x, ball.y, pose.x, pose.y)
        candidates = [(
            ball.x + math.cos(outward) * radius,
            ball.y + math.sin(outward) * radius,
        )]
        for x in (-half_l, half_l):
            remaining = radius ** 2 - (x - ball.x) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((x, ball.y + sign * offset) for sign in (-1.0, 1.0))
        for y in (-half_w, half_w):
            remaining = radius ** 2 - (y - ball.y) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((ball.x + sign * offset, y) for sign in (-1.0, 1.0))
        candidates = [
            t for t in candidates
            if t == clamp_inside_field(context, *t, margin=OPP_SET_PLAY_FIELD_MARGIN_M)
        ]
        if not candidates:
            p.stop()
            continue

        px, py = pose.x, pose.y
        target = min(candidates, key=lambda t: dist(px, py, *t))
        # Short omnidirectional steps follow the retreat line directly.
        # Do not invoke path avoidance: the movement is away from the ball.
        length = dist(pose.x, pose.y, *target)
        ratio = min(1.0, OMNI_DIST / (2.0 * length))
        step = (
            pose.x + (target[0] - pose.x) * ratio,
            pose.y + (target[1] - pose.y) * ratio,
        )
        p.action = "opp_set:retreat"
        p.walk_to(step, face=angle_to(pose.x, pose.y, ball.x, ball.y))


def _act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to its READY slot, assigned by player id.

    The slot comes from ``tactics.ready_slots`` rather than the player list
    index, so penalising one robot cannot leave the others in each other's
    spots and draw an illegal-positioning penalty.
    """
    slots = tactics.ready_slots(context, [p.id for p in players])
    for p in players:
        target = slots.get(p.id)
        p.action = "ready"
        if target is None:
            p.stop()
            continue
        p.walk_to(
            target,
            face=0.0,
            avoid_ball=True,
            avoid_robots=True,
        )



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates in red, using cubes while kicking and spheres otherwise.

    This runs for every player each frame regardless of phase, penalty, or
    readiness. The label contains the player ID and current high-level action
    (``p.action``), with ``[KICK]`` appended while kicking. Shape also indicates
    whether the player is in the kicking state.
    """
    from .framework import debugdraw

    pose = p.pose
    if pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(pose.x, pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(pose.x, pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(pose.x, pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Calculate player-to-ball distances and draw them each frame.

    This no longer depends on the analysis module; distances use the ball's
    current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Nothing can be drawn when the ball is not visible.
    if ball is None:
        return

    # 1. Draw the ball's current position as a green point.
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Draw distance labels: red for teammates and blue for opponents.
    for p in players:
        pose = p.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        pose = r.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )
