"""Manim animations for the Hungarian election simulation.

Run individual scenes:
    manim -pql animations.py HemicycleAnimation
    manim -pql animations.py SimulationConvergenceAnimation
    manim -pql animations.py TransferMatrixAnimation
    manim -pql animations.py DhondtAnimation

For high quality (slow):
    manim -pqh animations.py HemicycleAnimation

All videos saved to media/videos/animations/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, Rectangle, Circle, Text, Tex, MathTex,
    Arrow, Line, Dot, Square, ArcPolygon,
    BLUE, ORANGE, RED, GREEN, GRAY, WHITE, BLACK, YELLOW,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Transform, Write, DrawBorderThenFill,
    Create, Uncreate, GrowFromCenter, ShrinkToCenter,
    ReplacementTransform, AnimationGroup, LaggedStart,
    Axes, BarChart, NumberLine,
    ValueTracker, always_redraw,
    PI, TAU, DEGREES,
    config, tempconfig,
)
from manim import VMobject, ParametricFunction, ManimColor

def Color(hex_or_val):
    """Compatibility shim: accept hex string or ManimColor."""
    if isinstance(hex_or_val, str):
        s = hex_or_val.lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return ManimColor(f"#{s}")
    return hex_or_val

# ── Project imports ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PARTY_HEX = {
    "Fidesz-KDNP": "#fd8204",
    "TISZA":        "#1B4D8E",
    "Baloldal":     "#e6331a",
    "Mi Hazank":    "#215b2c",
    "MKKP":         "#888888",
    "German":       "#c0a060",
}

# Fixed median seats for default animation (from baseline sim)
MEDIAN_SEATS = {
    "TISZA":        112,
    "Fidesz-KDNP":  79,
    "Mi Hazank":    6,
    "Baloldal":     0,
    "MKKP":         0,
    "German":       1,   # nationality seat
}
TOTAL_SEATS = 199


# ── Helpers ───────────────────────────────────────────────────────────────────

def _party_color(name: str) -> Color:
    hex_str = PARTY_HEX.get(name, "#aaaaaa")
    return Color(hex_str)


def _hemicycle_positions(n_seats: int, n_rows: int = 8) -> np.ndarray:
    """Compute (x, y) positions for n_seats arranged in a hemicycle arc."""
    seats_per_row = np.linspace(n_seats // (n_rows + 1), n_seats // (n_rows - 2), n_rows).astype(int)
    seats_per_row[-1] += n_seats - seats_per_row.sum()

    positions = []
    for row_idx, count in enumerate(seats_per_row):
        r = 1.5 + row_idx * 0.4
        angles = np.linspace(PI * 0.05, PI * 0.95, count)
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a) - 0.5, 0])

    return np.array(positions[:n_seats])


# ── Scene 1: Parliament hemicycle ─────────────────────────────────────────────

class HemicycleAnimation(Scene):
    """Animates the 199-seat parliament hemicycle filling up by party."""

    def construct(self):
        # Title
        title = Text("2026 Hungarian Election — Seat Projection",
                     font_size=28, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Build seat party sequence (biggest party blocks first for visual clarity)
        party_order = ["TISZA", "Fidesz-KDNP", "Mi Hazank", "Baloldal", "MKKP", "German"]
        seat_sequence = []
        for p in party_order:
            seat_sequence.extend([p] * MEDIAN_SEATS.get(p, 0))
        # Pad to 199 if needed
        while len(seat_sequence) < TOTAL_SEATS:
            seat_sequence.append("MKKP")

        positions = _hemicycle_positions(TOTAL_SEATS, n_rows=9)

        # Scale positions to fit screen
        scale = 1.1
        cx, cy = 0.0, -2.0
        dots = VGroup()
        dot_objects = []

        for i, party in enumerate(seat_sequence):
            x, y, _ = positions[i] * scale
            dot = Dot(point=[cx + x, cy + y, 0],
                      radius=0.095,
                      color=_party_color(party))
            dot_objects.append(dot)
            dots.add(dot)

        # Majority line
        maj_line = Line(
            start=[cx - 0.01, cy, 0],
            end=[cx - 0.01, cy + 4.5, 0],
            color=RED,
            stroke_width=2,
        )
        maj_label = Text("Majority\n(100 seats)", font_size=16, color=RED).next_to(
            maj_line, RIGHT, buff=0.1
        ).shift(UP * 1.5)

        # Animate: draw dots group by group per party
        self.play(Create(maj_line), Write(maj_label), run_time=0.8)

        seat_counter = ValueTracker(0)
        counter_text = always_redraw(lambda: Text(
            f"{int(seat_counter.get_value())} / {TOTAL_SEATS}",
            font_size=20, color=GRAY,
        ).to_corner(DOWN + RIGHT, buff=0.4))
        self.add(counter_text)

        offset = 0
        for party in party_order:
            n = MEDIAN_SEATS.get(party, 0)
            if n == 0:
                continue
            group_dots = VGroup(*dot_objects[offset:offset + n])
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in group_dots],
                             lag_ratio=0.03),
                seat_counter.animate.set_value(offset + n),
                run_time=max(0.5, n * 0.015),
            )
            offset += n

        # Party legend
        legend_items = VGroup()
        for i, party in enumerate(p for p in party_order if MEDIAN_SEATS.get(p, 0) > 0):
            n = MEDIAN_SEATS[party]
            dot = Dot(radius=0.12, color=_party_color(party))
            lbl = Text(f"{party}: {n}", font_size=16, color=WHITE)
            item = VGroup(dot, lbl).arrange(RIGHT, buff=0.15)
            legend_items.add(item)
        legend_items.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        legend_items.to_corner(DOWN + LEFT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(item) for item in legend_items], lag_ratio=0.15))

        # Final annotation
        outcome = Text(
            f"Median: TISZA {MEDIAN_SEATS['TISZA']} seats  |  Fidesz {MEDIAN_SEATS['Fidesz-KDNP']} seats",
            font_size=22, color=YELLOW,
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(outcome))
        self.wait(3)


# ── Scene 2: Simulation convergence ───────────────────────────────────────────

class SimulationConvergenceAnimation(Scene):
    """Shows the TISZA seat distribution building up as MC draws are added."""

    def construct(self):
        title = Text("MC Simulation Convergence: TISZA Seat Distribution",
                     font_size=24, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Load actual simulation results if available, otherwise synthesise
        try:
            import json
            rpt_path = Path("output/reports/simulation_report.json")
            if rpt_path.exists():
                with open(rpt_path) as f:
                    rpt = json.load(f)
                med_t = rpt["seat_medians"]["TISZA"]["total"]
                std_t = (rpt["seat_medians"]["TISZA"]["p95"]
                         - rpt["seat_medians"]["TISZA"]["p5"]) / 3.28
            else:
                raise FileNotFoundError
        except Exception:
            med_t, std_t = 112, 22

        rng = np.random.default_rng(42)
        all_draws = rng.normal(med_t, std_t, 1000).clip(0, 199).astype(int)

        # Draw axes
        ax = Axes(
            x_range=[0, 199, 20],
            y_range=[0, 0.06, 0.01],
            axis_config={"color": WHITE},
            x_length=10,
            y_length=4,
            tips=False,
        ).shift(DOWN * 0.5)
        x_label = ax.get_x_axis_label(Text("TISZA seats", font_size=18))
        y_label = ax.get_y_axis_label(Text("Density", font_size=18), edge=LEFT)
        self.play(Create(ax), Write(x_label), Write(y_label))

        # Majority line
        maj_x = ax.c2p(100, 0)
        maj_top = ax.c2p(100, 0.06)
        maj_line = Line(maj_x, maj_top, color=RED, stroke_width=2)
        maj_lbl = Text("100 (majority)", font_size=14, color=RED).next_to(maj_top, UP, buff=0.05)
        self.play(Create(maj_line), Write(maj_lbl))

        # Animate histogram building up
        batch_sizes = [10, 20, 50, 100, 200, 500, 1000]
        prev_bars = None

        for n in batch_sizes:
            draws_n = all_draws[:n]
            counts, edges = np.histogram(draws_n, bins=30, range=(0, 199), density=True)
            bar_group = VGroup()
            for i, (cnt, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:])):
                if cnt > 0:
                    bot = ax.c2p(lo, 0)
                    top = ax.c2p(hi, cnt)
                    h = abs(top[1] - bot[1])
                    w = abs(top[0] - ax.c2p(lo, 0)[0])
                    bar = Rectangle(
                        width=w, height=h,
                        fill_color=Color(PARTY_HEX["TISZA"]),
                        fill_opacity=0.7,
                        stroke_width=0,
                    ).move_to([(bot[0] + top[0]) / 2, (bot[1] + top[1]) / 2, 0])
                    bar_group.add(bar)

            count_lbl = Text(f"n = {n}", font_size=22, color=YELLOW).to_corner(UP + RIGHT, buff=0.5)

            if prev_bars is None:
                self.play(FadeIn(bar_group), Write(count_lbl), run_time=0.6)
            else:
                self.play(
                    ReplacementTransform(prev_bars, bar_group),
                    ReplacementTransform(old_lbl, count_lbl),
                    run_time=0.7,
                )
            prev_bars = bar_group
            old_lbl = count_lbl
            self.wait(0.4)

        # Final median line
        med_x = ax.c2p(med_t, 0)
        med_top = ax.c2p(med_t, 0.055)
        med_line = Line(med_x, med_top, color=YELLOW, stroke_width=2.5)
        med_lbl = Text(f"Median: {med_t}", font_size=16, color=YELLOW).next_to(med_top, UP, buff=0.05)
        self.play(Create(med_line), Write(med_lbl))
        self.wait(3)


# ── Scene 3: Transfer matrix ──────────────────────────────────────────────────

class TransferMatrixAnimation(Scene):
    """Visualises the calibrated voter transfer matrix as an animated grid."""

    def construct(self):
        # Try to load calibrated Q from a quick calibration
        try:
            from lib.transfer_model import (
                load_baseline_and_matrix, load_nowcast,
                calibrate_transfer_matrix, Q_PRIOR, SOURCES, TARGETS,
            )
            from lib.config import SimConfig
            cfg = SimConfig()
            V22, _ = load_baseline_and_matrix()
            nowcast_shares, nowcast_se, _ = load_nowcast(cfg.nowcast_json)
            Q = calibrate_transfer_matrix(V22, nowcast_shares, nowcast_se, Q_PRIOR, cfg.lam_prior)
            sources = SOURCES
            targets = TARGETS
        except Exception:
            # Fall back to prior if data not available
            from lib.transfer_model import Q_PRIOR as Q, SOURCES as sources, TARGETS as targets

        title = Text("Calibrated Voter Transfer Matrix",
                     font_size=26, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Write(title))
        subtitle = Text("2022 source → 2026 target (each row sums to 100%)",
                        font_size=18, color=GRAY).next_to(title, DOWN, buff=0.1)
        self.play(FadeIn(subtitle))

        rows, cols = Q.shape
        cell_w, cell_h = 1.05, 0.72
        grid_w = cols * cell_w
        grid_h = rows * cell_h
        origin = np.array([-grid_w / 2, grid_h / 2 - cell_h / 2, 0]) + DOWN * 0.4

        # Column headers
        col_labels = VGroup()
        for j, tgt in enumerate(targets):
            lbl = Text(tgt.replace("26", ""), font_size=13, color=YELLOW)
            pos = origin + RIGHT * (j * cell_w + cell_w / 2) + UP * cell_h
            lbl.move_to(pos)
            col_labels.add(lbl)
        self.play(FadeIn(col_labels))

        # Row labels
        row_labels = VGroup()
        for i, src in enumerate(sources):
            lbl = Text(src.replace("22", ""), font_size=13, color=YELLOW)
            pos = origin + LEFT * 0.6 + DOWN * (i * cell_h)
            lbl.move_to(pos)
            row_labels.add(lbl)
        self.play(FadeIn(row_labels))

        # Animate cells row by row
        all_cells = VGroup()
        for i, src in enumerate(sources):
            row_cells = VGroup()
            for j in range(cols):
                val = Q[i, j]
                intensity = val  # 0–1
                # Interpolate: low = dark blue, high = bright orange/red
                r = int(255 * min(1.0, intensity * 2.0))
                g = int(255 * max(0.0, 0.6 - intensity))
                b = int(255 * max(0.0, 0.8 - intensity * 1.5))
                cell_color = Color(f"#{r:02x}{g:02x}{b:02x}")

                pos = origin + RIGHT * (j * cell_w) + DOWN * (i * cell_h)
                rect = Rectangle(
                    width=cell_w * 0.9, height=cell_h * 0.88,
                    fill_color=cell_color, fill_opacity=0.9,
                    stroke_color=WHITE, stroke_width=0.5,
                ).move_to(pos)

                pct_val = val * 100
                txt_color = WHITE if pct_val > 30 else WHITE
                txt = Text(f"{pct_val:.0f}%", font_size=14, color=txt_color)
                txt.move_to(pos)

                cell_group = VGroup(rect, txt)
                row_cells.add(cell_group)
                all_cells.add(cell_group)

            self.play(
                LaggedStart(*[FadeIn(c) for c in row_cells], lag_ratio=0.08),
                run_time=0.7,
            )

        # Highlight the most sensitive cells
        highlights = [
            (0, 0, "Fidesz retention"),
            (1, 1, "Opp→TISZA transfer"),
            (0, 6, "Fidesz→Abstain"),
        ]

        self.wait(0.5)
        for i, j, label in highlights:
            cell_idx = i * cols + j
            cell = all_cells[cell_idx]
            highlight = Rectangle(
                width=cell_w * 0.9, height=cell_h * 0.88,
                fill_opacity=0, stroke_color=YELLOW, stroke_width=3,
            ).move_to(cell)
            annot = Text(label, font_size=14, color=YELLOW).next_to(highlight, DOWN, buff=0.08)
            self.play(Create(highlight), Write(annot), run_time=0.5)
            self.wait(0.8)
            self.play(FadeOut(highlight), FadeOut(annot), run_time=0.4)

        self.wait(2)


# ── Scene 4: d'Hondt allocation step-by-step ──────────────────────────────────

class DhondtAnimation(Scene):
    """Step-by-step animation of the d'Hondt list seat allocation."""

    def construct(self):
        title = Text("d'Hondt List Seat Allocation (92 party seats)",
                     font_size=24, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Illustrative vote totals (in millions, 2026 projection)
        parties = ["Fidesz-KDNP", "TISZA", "Mi Hazank", "Other"]
        votes   = [2_100_000,      2_500_000,  320_000,   180_000]
        colors  = [PARTY_HEX[p] if p in PARTY_HEX else "#aaa" for p in parties]

        # Show vote bars
        ax = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 3_000_000, 500_000],
            x_length=7,
            y_length=3.5,
            tips=False,
        ).shift(LEFT * 1.5 + DOWN * 0.5)
        bar_labels = [Text(p.replace("-KDNP", "").replace(" ", "\n"), font_size=14)
                      for p in parties]
        self.play(Create(ax))

        bar_rects = VGroup()
        for i, (party, v, c) in enumerate(zip(parties, votes, colors)):
            top  = ax.c2p(i + 0.5, v)
            bot  = ax.c2p(i + 0.5, 0)
            h = abs(top[1] - bot[1])
            bar = Rectangle(
                width=0.6, height=h,
                fill_color=Color(c), fill_opacity=0.8, stroke_width=0,
            ).move_to([(top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2, 0])
            bar_rects.add(bar)

        self.play(LaggedStart(*[GrowFromCenter(b) for b in bar_rects], lag_ratio=0.2), run_time=1.5)

        # Iteratively award seats
        seats_won = {p: 0 for p in parties}
        seat_display = VGroup()
        seat_col_x = 4.0

        seat_header = Text("Seats won:", font_size=18, color=YELLOW).move_to(
            [seat_col_x, 2.5, 0]
        )
        self.play(Write(seat_header))

        seat_texts = {
            p: always_redraw(lambda p=p: Text(
                f"{p.replace('-KDNP','').replace(' ', ' ')}: {seats_won[p]}",
                font_size=16, color=Color(PARTY_HEX.get(p, '#aaa')),
            ).move_to([seat_col_x, 2.0 - list(parties).index(p) * 0.5, 0]))
            for p in parties
        }
        for t in seat_texts.values():
            self.add(t)

        for round_num in range(1, 13):
            # Compute quotients
            quotients = [(votes[i] / (seats_won[p] + 1), i, p)
                         for i, p in enumerate(parties)]
            best_q, best_i, best_p = max(quotients, key=lambda x: x[0])

            # Highlight winner
            winner_bar = bar_rects[best_i]
            flash = Rectangle(
                width=0.65, height=winner_bar.height + 0.05,
                fill_opacity=0, stroke_color=YELLOW, stroke_width=3,
            ).move_to(winner_bar)
            round_txt = Text(
                f"Round {round_num}: {best_p.replace('-KDNP','')} "
                f"(quotient {best_q/1e6:.2f}M)",
                font_size=15, color=WHITE,
            ).to_edge(DOWN, buff=0.4)

            self.play(Create(flash), Write(round_txt), run_time=0.5)

            seats_won[best_p] += 1
            self.play(FadeOut(flash), FadeOut(round_txt), run_time=0.3)

        self.wait(2)
