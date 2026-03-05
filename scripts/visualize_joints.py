#!/usr/bin/env python3
"""
Terminal visualization of robot joint torques and temperatures.

As a library:
    from scripts.visualize_joints import print_observations, monitor_robot

    print_observations(robot)   # single snapshot
    monitor_robot(robot)        # live-updating loop, Ctrl-C to stop
"""

import time

import numpy as np

try:
    from rich import box as rbox
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

JOINT_NAMES = ["J1  base", "J2  shoulder", "J3  elbow", "J4  wrist1", "J5  wrist2", "J6  gripper"]
BAR_WIDTH = 28
TEMP_WARN = 50.0   # °C — yellow
TEMP_CRIT = 70.0   # °C — red


# ── helpers ───────────────────────────────────────────────────────────────────
def _motor_torque_limits(robot: object) -> np.ndarray:
    from i2rt.motor_drivers.utils import MotorType
    return np.array([MotorType.get_motor_constants(mt).TORQUE_MAX for _, mt in robot.motor_chain.motor_list])


def _color_for_ratio(ratio: float) -> str:
    if ratio < 0.35:
        return "green"
    if ratio < 0.65:
        return "yellow"
    return "red"


def _color_for_temp(temp: float) -> str:
    if temp < TEMP_WARN:
        return "green"
    if temp < TEMP_CRIT:
        return "yellow"
    return "bold red"


def _rich_bar(value: float, max_val: float, width: int = BAR_WIDTH) -> "Text":
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    bar = Text()
    bar.append("█" * filled, style=_color_for_ratio(ratio))
    bar.append("░" * (width - filled), style="dim")
    return bar


def _ansi_bar(value: float, max_val: float, width: int = BAR_WIDTH) -> str:
    ANSI = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "dim": "\033[2m", "reset": "\033[0m"}
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    color = ANSI[_color_for_ratio(ratio)]
    return f"{color}{'█' * filled}{ANSI['reset']}{ANSI['dim']}{'░' * (width - filled)}{ANSI['reset']}"


# ── table builders ────────────────────────────────────────────────────────────
def _make_torque_table(obs: dict, limits: np.ndarray) -> "Table":
    joint_eff = np.asarray(obs["joint_eff"])
    n = len(joint_eff)

    unique = np.unique(limits)
    title_scale = f"±{unique[0]:.0f} Nm" if len(unique) == 1 else "per-motor limits"

    table = Table(
        box=rbox.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        title=f"[bold]Joint Torques[/]  ({title_scale})",
        title_style="white",
        expand=False,
    )
    table.add_column("Joint", style="bold white", width=15, no_wrap=True)
    table.add_column("Load", width=BAR_WIDTH + 2, no_wrap=True)
    table.add_column("Torque (Nm)", width=12, justify="right", no_wrap=True)
    table.add_column("% max", width=6, justify="right", no_wrap=True)

    peak_idx = int(np.argmax(np.abs(joint_eff)))

    for i in range(n):
        base_name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i + 1}"
        name = f"[bold yellow]▶ {base_name}[/]" if i == peak_idx else base_name
        eff = joint_eff[i]
        lim = limits[i]
        pct = abs(eff) / lim * 100
        color = _color_for_ratio(abs(eff) / lim)
        sign = "+" if eff >= 0 else ""
        table.add_row(
            name,
            _rich_bar(eff, lim),
            f"[{color}]{sign}{eff:.4f}[/]",
            f"[{color}]{pct:5.1f}%[/]",
        )

    return table


def _make_temp_table(obs: dict) -> "Table":
    temp_mos = np.asarray(obs["temp_mos"])
    temp_rotor = np.asarray(obs["temp_rotor"])
    n = len(temp_mos)

    hot_joints = np.where((temp_mos >= TEMP_CRIT) | (temp_rotor >= TEMP_CRIT))[0]
    warning = " [bold red]⚠ OVERHEATING[/]" if len(hot_joints) > 0 else ""

    table = Table(
        box=rbox.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        title=f"[bold]Temperatures[/]{warning}",
        title_style="white",
        expand=False,
        caption=f"[dim]warn >{TEMP_WARN:.0f}°C   crit >{TEMP_CRIT:.0f}°C[/]",
    )
    table.add_column("Joint", style="bold white", width=15, no_wrap=True)
    table.add_column("MOS (°C)", width=10, justify="right", no_wrap=True)
    table.add_column("Rotor (°C)", width=11, justify="right", no_wrap=True)

    for i in range(n):
        base_name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i + 1}"
        mos = temp_mos[i]
        rotor = temp_rotor[i]
        hot = (mos >= TEMP_CRIT) or (rotor >= TEMP_CRIT)
        name = f"[bold red]▶ {base_name}[/]" if hot else base_name
        table.add_row(
            name,
            f"[{_color_for_temp(mos)}]{mos:.0f}°C[/]",
            f"[{_color_for_temp(rotor)}]{rotor:.0f}°C[/]",
        )

    return table


def _make_renderable(obs: dict, limits: np.ndarray) -> object:
    torque_table = _make_torque_table(obs, limits)
    if "temp_mos" in obs and "temp_rotor" in obs:
        return Group(torque_table, _make_temp_table(obs))
    return torque_table


# ── public API ────────────────────────────────────────────────────────────────
def print_observations(robot: object) -> None:
    """Print a single snapshot of joint torques and temperatures."""
    obs = robot.get_observations()
    limits = _motor_torque_limits(robot)

    if HAS_RICH:
        import shutil
        cols = shutil.get_terminal_size((120, 24)).columns
        Console(width=cols).print(_make_renderable(obs, limits))
    else:
        _print_ansi(obs, limits)


def _print_ansi(obs: dict, limits: np.ndarray) -> None:
    joint_eff = np.asarray(obs["joint_eff"])
    n = len(joint_eff)
    print("\n  Joint Torques")
    print("  " + "─" * 72)
    peak_idx = int(np.argmax(np.abs(joint_eff)))
    for i in range(n):
        name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i + 1}"
        eff = joint_eff[i]
        lim = limits[i]
        pct = abs(eff) / lim * 100
        bar = _ansi_bar(eff, lim)
        sign = "+" if eff >= 0 else ""
        star = " ◀ peak" if i == peak_idx else ""
        print(f"  {name:<14} {bar}  {sign}{eff:8.4f} Nm  {pct:5.1f}% of ±{lim:.0f}{star}")

    if "temp_mos" in obs and "temp_rotor" in obs:
        temp_mos = np.asarray(obs["temp_mos"])
        temp_rotor = np.asarray(obs["temp_rotor"])
        ANSI_RED, ANSI_RESET = "\033[31m", "\033[0m"
        print(f"\n  Temperatures  (warn >{TEMP_WARN:.0f}°C  crit >{TEMP_CRIT:.0f}°C)")
        print("  " + "─" * 72)
        for i in range(n):
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i + 1}"
            mos, rotor = temp_mos[i], temp_rotor[i]
            hot = (mos >= TEMP_CRIT) or (rotor >= TEMP_CRIT)
            warn = f"{ANSI_RED} ⚠{ANSI_RESET}" if hot else ""
            print(f"  {name:<14}  MOS {mos:5.0f}°C   Rotor {rotor:5.0f}°C{warn}")
    print()


def monitor_robot(robot: object, update_hz: float = 10.0) -> None:
    """Live-updating terminal view of joint torques and temperatures. Press Ctrl-C to stop."""
    limits = _motor_torque_limits(robot)
    interval = 1.0 / update_hz

    if HAS_RICH:
        import shutil
        cols = shutil.get_terminal_size((120, 24)).columns
        console = Console(width=cols)
        with Live(console=console, refresh_per_second=update_hz, screen=False) as live:
            live.update("[dim]Connecting…[/]")
            try:
                while True:
                    obs = robot.get_observations()
                    live.update(_make_renderable(obs, limits))
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass
    else:
        import os
        try:
            while True:
                obs = robot.get_observations()
                os.system("clear")
                _print_ansi(obs, limits)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


# ── demo with fake data ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running with fake observations (no robot needed).")
    print("Press Ctrl-C to stop.\n")

    class FakeMotorChain:
        motor_list = [
            (1, "DM4310"), (2, "DM4310"), (3, "DM4310"),
            (4, "DM4310"), (5, "DM4310"), (6, "DM3507"),
        ]

    class FakeRobot:
        motor_chain = FakeMotorChain()

        def __init__(self) -> None:
            self._t = 0.0

        def get_observations(self) -> dict:
            self._t += 0.05
            eff = np.array([
                np.sin(self._t * 0.3) * 0.5,
                np.sin(self._t * 0.7 + 1.0) * 4.0,
                np.sin(self._t * 1.1 + 2.0) * 8.5,
                np.sin(self._t * 0.9 + 0.5) * 3.0,
                np.cos(self._t * 1.3) * 0.3,
                np.cos(self._t * 0.5) * 1.5,
            ])
            temp_mos   = np.array([24., 25., 28., 30., 27., 26.]) + np.sin(self._t * 0.1) * 5
            temp_rotor = np.array([21., 23., 29., 31., 25., 25.]) + np.sin(self._t * 0.1) * 5
            return {
                "joint_eff": eff,
                "joint_pos": np.zeros(6),
                "joint_vel": np.zeros(6),
                "temp_mos": temp_mos,
                "temp_rotor": temp_rotor,
            }

    monitor_robot(FakeRobot(), update_hz=10.0)