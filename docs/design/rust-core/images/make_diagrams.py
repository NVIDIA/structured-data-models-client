# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generates the Rust core migration plan diagrams: architecture.svg, pql.svg and delivery.svg.

The SVGs are written next to this script. To regenerate them and their PNGs:

    python make_diagrams.py
    for f in architecture pql delivery; do rsvg-convert -z 2 $f.svg -o $f.png; done

Colors: orange is the Rust core, blue is each language, grey is outside sdm-client.
"""

from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent

ORANGE, BLUE, GREY = "#c9781c", "#3d72b8", "#666666"
ORANGE_FILL, BLUE_FILL, GREY_FILL = "#fdf3e6", "#eaf2fc", "#f7f7f7"
MARK = {ORANGE: "o", BLUE: "b", GREY: "g"}


class Canvas:
    """One SVG: boxes, bands and arrows, with labels drawn last so they sit over the arrows."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.out = []
        self.late = []

    def title(self, text, subtitle=None):
        self.out.append(f'<text x="{self.w / 2}" y="38" font-size="22" font-weight="bold" text-anchor="middle" '
                        f'fill="#111">{escape(text)}</text>')
        if subtitle:
            self.out.append(f'<text x="{self.w / 2}" y="64" font-size="14" text-anchor="middle" fill="#444">'
                            f'{escape(subtitle)}</text>')

    def note(self, x, y, text, size=13, bold=False, color="#444"):
        weight = ' font-weight="bold"' if bold else ""
        self.out.append(f'<text x="{x}" y="{y}" font-size="{size}"{weight} fill="{color}">{escape(text)}</text>')

    def text_lines(self, x, y, lines, size=14, anchor="middle", bold=1):
        for i, line in enumerate(lines):
            weight = "bold" if i < bold else "normal"
            self.out.append(f'<text x="{x}" y="{y + i * (size + 5)}" font-size="{size}" font-weight="{weight}" '
                            f'text-anchor="{anchor}" fill="#222">{escape(line)}</text>')

    def box(self, x1, y1, x2, y2, lines, fill="#ffffff", stroke="#777", shape="rect", size=14, bold=1,
            dashed=False):
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        if shape == "cyl":
            rx, ry = (x2 - x1) / 2, 9
            self.out.append(f'<path d="M{x1},{y1 + ry} A{rx},{ry} 0 0,1 {x2},{y1 + ry} L{x2},{y2 - ry} '
                            f'A{rx},{ry} 0 0,1 {x1},{y2 - ry} Z" fill="{fill}" stroke="{stroke}"/>')
            self.out.append(f'<path d="M{x1},{y1 + ry} A{rx},{ry} 0 0,0 {x2},{y1 + ry}" fill="none" '
                            f'stroke="{stroke}"/>')
        elif shape == "diamond":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.out.append(f'<path d="M{cx},{y1} L{x2},{cy} L{cx},{y2} L{x1},{cy} Z" fill="{fill}" '
                            f'stroke="{stroke}" stroke-width="1.3"/>')
        else:
            self.out.append(f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="10" '
                            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
        n = len(lines)
        cy = (y1 + y2) / 2 - (n - 1) * (size + 5) / 2 + 5
        self.text_lines((x1 + x2) / 2, cy, lines, size=size, bold=bold)

    def band(self, x1, y1, x2, y2, title, fill, stroke, dashed=False):
        dash = ' stroke-dasharray="8,5"' if dashed else ""
        self.out.append(f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="14" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
        halo = f'stroke="{fill}" stroke-width="6" stroke-linejoin="round" paint-order="stroke"'
        self.late.append(f'<text x="{x1 + 16}" y="{y1 + 24}" font-size="16" font-weight="bold" fill="{stroke}" '
                         f'{halo}>{escape(title)}</text>')

    def arrow(self, points, color=GREY, both=False, label=None, lx=None, ly=None, anchor="start", width=2.2,
              dashed=False, head=True):
        m = MARK[color]
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        start = f' marker-start="url(#s{m})"' if both else ""
        end = f' marker-end="url(#e{m})"' if head else ""
        dash = ' stroke-dasharray="7,5"' if dashed else ""
        self.out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{end}{start}{dash}/>')
        if label:
            self.late.append(f'<text x="{lx}" y="{ly}" font-size="14" fill="{color}" text-anchor="{anchor}" '
                             f'font-style="italic" stroke="#ffffff" stroke-opacity="0.9" stroke-width="3" '
                             f'paint-order="stroke">{escape(label)}</text>')

    def save(self, name):
        defs = []
        for key, color in (("o", ORANGE), ("b", BLUE), ("g", GREY)):
            defs.append(f'<marker id="e{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
            defs.append(f'<marker id="s{key}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" '
                        f'markerHeight="7" orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{color}"/></marker>')
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="Helvetica, Arial, sans-serif">'
               f'<defs>{"".join(defs)}</defs><rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
               + "".join(self.out) + "".join(self.late) + "</svg>\n")
        (HERE / name).write_text(svg)


def architecture():
    c = Canvas(1600, 1075)
    c.title("New plan: one Rust core behind a typed contract",
            "The core decides what data it needs; each language fetches it, converts it to Arrow, "
            "and shows the result its own way.")

    # Bands. The layout keeps every vertical arrow clear of the band titles.
    c.band(30, 86, 1570, 204, "Outside sdm-client", GREY_FILL, "#999999", dashed=True)
    c.band(30, 232, 1570, 416, "Python and Java: every query and network call happens here", BLUE_FILL, BLUE)
    c.band(30, 556, 1570, 900, "Rust core: shared logic, no database or network access", ORANGE_FILL, ORANGE)

    # Outside
    c.box(690, 122, 950, 190, ["Model endpoint", "NIM or Databricks Model Serving"])
    c.box(970, 118, 1210, 194, ["Databases", "Databricks, Snowflake, Postgres, ..."], shape="cyl")
    c.box(1230, 122, 1410, 190, ["Other data sources", "e.g. Pyramid"])

    # Each language
    c.box(60, 278, 300, 398, ["User API", "Schema.from_source / from_data", "PredictiveQuery.materialize,",
                              "predict"], size=13)
    c.box(330, 278, 600, 398, ["Errors and results", "maps error kinds to native exceptions;",
                               "turns Arrow results into pandas", "or Java objects"], size=13)
    c.box(700, 278, 940, 398, ["Model transport", "HTTP(S)"])
    c.box(970, 278, 1300, 398, ["Data providers", "SQL connector: runs the core's SQL", "with the vendor driver",
                                "custom provider: answers data", "requests itself, e.g. Pyramid"])
    c.box(1330, 278, 1540, 398, ["Type conversion", "native types to Arrow,", "through a reviewed table:",
                                 "pandas in Python,", "JDBC in Java"], size=13)

    # The typed contract between the language and the core
    c.out.append('<line x1="30" y1="500" x2="1570" y2="500" stroke="#555" stroke-width="1.2" '
                 'stroke-dasharray="3,3"/>')
    c.box(1090, 472, 1410, 528, ["Typed contract", "Arrow types, typed operations, error kinds;",
                                 "no calls back into the language"], stroke="#555", size=13)

    # Rust core: the entry point in the middle, the services around it
    c.box(450, 604, 600, 850, ["Core operations", "typed operations,", "run loop"], size=13)
    c.box(60, 604, 420, 704, ["PQL", "from the shared PQL crate:", "parser, validator, error codes"])
    c.box(60, 750, 420, 850, ["Labels"])
    c.box(740, 604, 940, 704, ["Request builder", "payload, response,", "result"], size=13)
    c.box(970, 604, 1540, 704, ["Data access", "typed data requests; built-in SQL dialects compile them to SQL;",
                                "capability flags; in-memory fallback"])
    c.box(970, 750, 1220, 850, ["Schema inference", "keys, time columns, links,", "over typed Arrow"], size=13)
    c.box(1260, 750, 1540, 850, ["Sampling", "plan, neighbor sampler,", "random streams"], size=13)

    # Outside <-> language
    c.arrow([(840, 278), (840, 190)], GREY, both=True, label="HTTP(S)", lx=848, ly=224)
    c.arrow([(1090, 278), (1090, 194)], GREY, both=True, label="SQL, rows", lx=1098, ly=224)
    c.arrow([(1260, 278), (1260, 190)], GREY, both=True, label="its own API", lx=1268, ly=224)

    # Inside the language: provider rows go through conversion
    c.arrow([(1300, 338), (1330, 338)], GREY)

    # Entry and exit
    c.arrow([(180, 398), (180, 450), (490, 450), (490, 604)], BLUE, label="1  query, Arrow tables", lx=192,
            ly=442)
    c.arrow([(575, 604), (575, 398)], ORANGE, label="7  typed result", lx=583, ly=532)

    # Data round trip
    c.arrow([(1050, 604), (1050, 398)], ORANGE, label="2  SQL, or a typed request", lx=1058, ly=444, width=2.8)
    c.arrow([(1440, 398), (1440, 604)], BLUE, label="3  rows as Arrow", lx=1432, ly=444, anchor="end", width=2.8)

    # Inside the core
    c.arrow([(450, 654), (420, 654)], GREY)
    c.arrow([(450, 800), (420, 800)], GREY)
    c.arrow([(525, 850), (525, 878), (1400, 878), (1400, 850)], GREY)
    c.arrow([(1095, 878), (1095, 850)], GREY)
    c.arrow([(1095, 750), (1095, 704)], GREY, both=True, label="catalog", lx=1103, ly=732)
    c.arrow([(1400, 750), (1400, 704)], GREY, both=True, label="data for each hop", lx=1408, ly=732)

    # Model request and response
    c.arrow([(600, 640), (740, 640)], GREY, label="4  samples, labels", lx=670, ly=630, anchor="middle")
    c.arrow([(740, 680), (600, 680)], GREY, label="result", lx=670, ly=670, anchor="middle")
    c.arrow([(780, 604), (780, 398)], ORANGE, label="5  request", lx=788, ly=444)
    c.arrow([(900, 398), (900, 604)], BLUE, label="6  response", lx=908, ly=444)

    c.note(60, 928, "Steps 2 and 3 repeat per table and per sampling hop.")

    # Example strip
    c.note(60, 966, "Example: who handles what", size=15, bold=True, color="#222")
    c.box(60, 980, 540, 1046, ["Rust core", "writes Databricks SQL, or a typed request for Pyramid"],
          fill=ORANGE_FILL, stroke=ORANGE, size=13)
    c.box(570, 980, 1040, 1046, ["Python", "runs it with databricks-sql-connector or Spark"],
          fill=BLUE_FILL, stroke=BLUE, size=13)
    c.box(1070, 980, 1540, 1046, ["Java", "runs it with the Databricks JDBC driver,", "or a Pyramid provider"],
          fill=BLUE_FILL, stroke=BLUE, size=13)

    c.save("architecture.svg")


def pql():
    c = Canvas(1300, 740)
    c.title("One PQL implementation, used everywhere (target)",
            "The grammar, parser and validator live once, in Rust; every consumer uses the same crate.")

    c.band(30, 90, 600, 530, "Shared PQL repository", ORANGE_FILL, ORANGE)
    c.box(70, 134, 560, 204, ["Grammar files", "token table for logos, LR(1) grammar for lalrpop"])
    c.box(70, 262, 560, 362, ["pql crate", "parser, validator, error codes and spans,",
                              "validated query as versioned JSON"])
    c.box(70, 430, 560, 500, ["Conformance suite", "shared valid and invalid queries, run by every consumer"])
    c.arrow([(315, 204), (315, 262)], GREY, label="builds the parser", lx=325, ly=238)
    c.arrow([(315, 430), (315, 362)], GREY, label="checks", lx=325, ly=401)

    # Consumers
    c.box(700, 130, 960, 210, ["sdm-client Rust core", "direct dependency"], fill=ORANGE_FILL, stroke=ORANGE)
    c.box(1020, 130, 1270, 210, ["sdm-client", "Python and Java"], fill=BLUE_FILL, stroke=BLUE, bold=2)
    c.box(700, 266, 960, 358, ["Kumo Python", "thin wheel over the crate", "(when Kumo adopts it)"],
          fill=BLUE_FILL, stroke=BLUE)
    c.box(700, 408, 960, 500, ["Kumo Java", "thin JNI jar", "(when Kumo adopts it)"], fill=BLUE_FILL, stroke=BLUE)

    c.arrow([(560, 312), (630, 312), (630, 170), (700, 170)], ORANGE)
    c.arrow([(630, 312), (700, 312)], ORANGE)
    c.arrow([(630, 312), (630, 454), (700, 454)], ORANGE)
    c.arrow([(960, 170), (1020, 170)], GREY, label="serves", lx=990, ly=160, anchor="middle")

    # Migration side box, checked against the same conformance queries
    c.box(70, 574, 560, 672, ["During migration", "today's ANTLR grammar kept as a CI reference until its",
                              "remaining users migrate; the first attempt's Rust parser",
                              "serves sdm-client as an interim parser until the shared",
                              "crate replaces it"],
          fill=GREY_FILL, stroke="#999999", size=13, dashed=True)
    c.arrow([(315, 500), (315, 574)], GREY, dashed=True, label="same queries, compared in CI", lx=325, ly=556)

    c.note(60, 712, "sdm-client adds only model capability checks on top of the validated query.")

    c.save("pql.svg")


def delivery():
    c = Canvas(1660, 610)
    c.title("Delivery: slices through the public API",
            "Each slice goes from the public API down to the core and merges to main at its gate.")

    top, bottom, mid = 198, 338, 268

    c.band(235, 156, 1345, 366, "S1 and S2 ship in Python and Java; S3's Java scope decided in R0", BLUE_FILL,
           BLUE)

    phases = [
        (60, 220, ["R0 Decide", "decisions, dates;", "bank the first", "attempt's best", "candidate"], 1),
        (250, 410, ["R1a Harness", "contract, test runners,", "CI guards;", "S0 PQL validation"], 1),
        (440, 600, ["R1b Spike", "DuckDB from_source", "to predict in", "both languages"], 1),
        (790, 950, ["R2", "S1 offline warehouses", "DuckDB, SQLite,", "Postgres"], 2),
        (980, 1140, ["R3", "S2 live warehouses", "Databricks, Snowflake,", "Lakebase;", "Python switches",
                     "warehouse paths"], 2),
        (1170, 1330, ["R4", "S3", "in-memory frames,", "ranking, forecasting,", "sessions"], 2),
        (1410, 1570, ["R5 Release", "delete old engine"], 1),
    ]
    for x1, x2, lines, bold in phases:
        c.box(x1, top, x2, bottom, lines, size=13, bold=bold)

    c.box(625, 223, 765, 313, ["owner decides", "go / no-go"], shape="diamond", size=12, bold=0)
    c.text_lines(695, 334, ["go: merge R1a", "and R1b to main"], size=12, bold=0)

    # Sequence arrows
    for x1, x2 in ((220, 250), (410, 440), (600, 625), (765, 790), (950, 980), (1140, 1170), (1330, 1410)):
        c.arrow([(x1, mid), (x2, mid)], GREY)

    # Merges to main: after the spike's go, after each slice gate, and at release
    main_y = 414
    c.out.append(f'<line x1="60" y1="{main_y}" x2="1570" y2="{main_y}" stroke="{GREY}" stroke-width="4"/>')
    c.note(60, main_y - 10, "main", size=14, bold=True, color=GREY)
    for x, y, label in ((777, mid, "merge to main"), (965, mid, "merge to main"), (1155, mid, "merge to main"),
                        (1355, mid, "merge to main"), (1490, bottom, "merge to main (release)")):
        c.arrow([(x, y), (x, main_y - 3)], ORANGE, width=2)
        c.out.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{ORANGE}"/>')
        c.note(x + 7, 392, label, size=12, color=ORANGE)

    # Optional side branch after S2
    c.arrow([(1155, mid), (1155, 118), (1210, 118)], GREY, dashed=True)
    c.box(1210, 92, 1450, 144, ["Pyramid provider", "(if in scope)"], fill=GREY_FILL, stroke="#999999",
          size=13, dashed=True)

    # Checks strip
    c.band(30, 450, 1630, 580, "Checks that run all along", GREY_FILL, "#999999")
    c.box(60, 490, 570, 560, ["Every PR", "unit, boundary guards, contract subset,",
                              "SQL goldens, provider kit, conversion tables"], size=13)
    c.box(600, 490, 1080, 560, ["Nightly", "full contract corpus, differential vs old release,", "performance"],
          size=13)
    c.box(1110, 490, 1600, 560, ["Weekly and before release", "weekly: live matrix",
                                 "before release: all layers, packaging and safety"], size=13)

    c.save("delivery.svg")


if __name__ == "__main__":
    architecture()
    pql()
    delivery()
