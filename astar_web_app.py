"""
A* SEARCH ALGORITHM - INTERACTIVE WEB VISUALIZER  (100% Python, built with Streamlit)
AI and Soft Computing - Academic Project

How to run (macOS Terminal, inside this folder):
    python3 -m pip install streamlit
    python3 astar_web_app.py          -> opens the website in your browser (http://localhost:8501)

Press "Start" once: the real A* search then plays automatically, step by step.
"""
import sys, math, time, html, heapq, inspect, subprocess
import streamlit as st


# ============================ 1. THE PROBLEM ============================
def define_graph():
    """Return the search graph, the heuristic table and the fixed drawing positions."""
    graph = {                                  # adjacency dictionary: node -> [(child, edge cost)]
        "S": [("A", 2), ("B", 3)],
        "A": [("C", 2), ("D", 4)],
        "B": [("E", 2), ("F", 5)],
        "C": [("H", 3)],
        "D": [("G", 3)],
        "E": [("G", 5)],                       # second, more expensive route to G
        "F": [("I", 2)],
        "G": [], "H": [], "I": [],
    }
    heur = {"S": 7, "A": 5, "B": 4, "C": 3, "D": 3, "E": 3,   # h(n): admissible and consistent
            "F": 3, "G": 0, "H": 4, "I": 2}
    pos = {"S": (0, 4.5),
           "A": (-2.4, 3), "B": (2.4, 3),
           "C": (-3.6, 1.5), "D": (-1.2, 1.5), "E": (1.2, 1.5), "F": (3.6, 1.5),
           "H": (-3.6, 0), "G": (0, 0), "I": (3.6, 0)}
    return graph, heur, pos


GRAPH, HEUR, POS = define_graph()
START, GOAL = "S", "G"


def heuristic(n):
    """h(n): estimated remaining cost from n to the goal."""
    return HEUR[n]


# ============================ 2. THE A* ALGORITHM ============================
def astar_search(start, goal):
    """Create all A* data structures. The search itself runs inside astar_steps()."""
    state = {"open": [(heuristic(start), start)],  # priority queue (min-heap) of (f, node)
          "closed": set(),                      # nodes already expanded
          "order": [],                          # same nodes, in expansion order (for display)
          "g": {start: 0},                      # best known actual cost from start
          "parent": {start: None},              # parent pointers to rebuild the path
          "path": []}
    state["steps"] = astar_steps(state, goal)         # generator: one real A* operation per next()
    return state


def astar_steps(state, goal):
    """The genuine A* loop. Every 'yield' pauses the algorithm so the animation can show it."""
    open_list, closed, g, parent = state["open"], state["closed"], state["g"], state["parent"]
    yield ("init", START, None)
    while open_list:
        f, n = heapq.heappop(open_list)         # node with the LOWEST f(n)
        if n in closed:                         # outdated duplicate entry -> ignore it
            yield ("stale", n, f)
            continue
        closed.add(n)                           # move n from OPEN to CLOSED
        state["order"].append(n)
        if n == goal:                           # stop when the goal is SELECTED
            yield ("goal", n, None)
            state["path"] = reconstruct_path(parent, goal)
            for i in range(1, len(state["path"])):  # highlight the path edge by edge
                yield ("path", state["path"][i], state["path"][i - 1])
            yield ("done", goal, None)
            return
        yield ("select", n, None)
        for nb, cost in GRAPH[n]:               # examine every neighbour of n
            if nb in closed:
                yield ("skip", n, nb)
                continue
            new_g = g[n] + cost                 # tentative cost of reaching nb through n
            if new_g < g.get(nb, math.inf):     # first time seen OR a cheaper route
                old_g = g.get(nb)
                g[nb], parent[nb] = new_g, n
                heapq.heappush(open_list, (new_g + heuristic(nb), nb))   # f = g + h
                yield ("update" if old_g is not None else "discover", n, (nb, old_g))
            else:
                yield ("worse", n, (nb, new_g))
        yield ("expanded", n, None)
    yield ("fail", None, None)


def reconstruct_path(parent, node):
    """Follow parent pointers back to the start, then reverse."""
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def open_entries(state):
    """Split heap entries into valid ones and outdated duplicates (for display)."""
    valid, stale = [], []
    for f, n in sorted(state["open"]):
        ok = n not in state["closed"] and f == state["g"][n] + heuristic(n)
        (valid if ok else stale).append((f, n))
    return valid, stale


def describe(ev, state, step):
    """Turn one algorithm event into panel text: (title, value lines, explanation)."""
    kind, n, info = ev
    g = state["g"]
    f = lambda x: g[x] + heuristic(x)
    vals = lambda x: f"g({x}) = {g[x]}   h({x}) = {heuristic(x)}   f({x}) = {f(x)}"
    title, lines = f"STEP {step} — EXPANDING NODE {n}", [f"Current node: {n}"]
    if kind == "ready":
        return ("A* SEARCH — READY",
                [f"Start node: {START}     Goal node: {GOAL}", "", "f(n) = g(n) + h(n)",
                 "g(n) = actual cost from S to n", "h(n) = estimated cost from n to G",
                 "f(n) = estimated total cost through n"],
                "Press START once. The whole search then plays automatically, step by step.")
    if kind == "init":
        title, lines = "STEP 0 — INITIALISATION", [f"Start node: {n}", vals(n)]
        expl = (f"A* starts at {n}. g({n}) = 0 because nothing has been paid yet, h({n}) = "
                f"{heuristic(n)} estimates the rest, so f({n}) = {f(n)}. {n} is pushed into OPEN.")
    elif kind in ("select", "goal"):
        lines.append(vals(n))
        ties = [m for fv, m in open_entries(state)[0] if fv == f(n)]
        tie = f" (Tie with {', '.join(ties)}: heapq picks the alphabetically smaller name.)" if ties else ""
        expl = (f"A* removes {n} from OPEN because f({n}) = {f(n)} is the lowest f(n) in the "
                f"priority queue.{tie} {n} moves to CLOSED and its neighbours are examined next.")
        if kind == "goal":
            title = f"STEP {step} — GOAL {n} SELECTED"
            expl = (f"The goal {n} now has the lowest f(n) = {f(n)} in OPEN.{tie} With an admissible "
                    f"heuristic no other path can be cheaper, so A* stops. Optimal cost = {g[n]}.")
    elif kind in ("discover", "update"):
        nb, old = info
        title = f"STEP {step} — EXPANDING {n}: " + (f"NEW NODE {nb}" if kind == "discover" else "BETTER PATH")
        lines = [f"Current node: {n}   g({n}) = {g[n]}",
                 f"g({nb}) = g({n}) + cost({n},{nb}) = {g[n]} + {g[nb] - g[n]} = {g[nb]}",
                 f"h({nb}) = {heuristic(nb)}",
                 f"f({nb}) = g({nb}) + h({nb}) = {g[nb]} + {heuristic(nb)} = {f(nb)}"]
        if kind == "discover":
            expl = (f"{nb} is seen for the first time. A* stores g({nb}) = {g[nb]}, sets "
                    f"parent({nb}) = {n} and pushes {nb} into OPEN with priority f = {f(nb)}.")
        else:
            expl = (f"Cheaper route found! Through {n}, g({nb}) drops from {old} to {g[nb]}. A* updates "
                    f"g({nb}), sets parent({nb}) = {n} and pushes a new entry with f = {f(nb)}. "
                    f"The old entry stays in the heap and will be ignored.")
    elif kind == "worse":
        nb, new_g = info
        lines.append(f"Route via {n}: g = {new_g}   known g({nb}) = {g[nb]}")
        expl = f"Reaching {nb} through {n} costs {new_g}, not cheaper than {g[nb]}, so it is ignored."
    elif kind == "skip":
        expl = f"{info} is already in CLOSED (fully expanded), so it is skipped."
    elif kind == "stale":
        title, lines = f"STEP {step} — IGNORING OLD ENTRY", [f"Popped entry: {n} (f = {info})"]
        expl = f"{n} is already closed with a cheaper cost, so this outdated heap entry is discarded."
    elif kind == "expanded":
        title = f"STEP {step} — {n} FULLY EXPANDED"
        expl = (f"All neighbours of {n} have been examined. {n} stays in CLOSED (explored). "
                f"Next, A* picks the OPEN node with the lowest f(n).")
    elif kind == "path":
        u, path = info, state["path"]
        title = "RECONSTRUCTING THE OPTIMAL PATH"
        lines = [f"parent({n}) = {u}", "Path: " + " → ".join(path[:path.index(n) + 1])]
        expl = (f"A* follows parent pointers back from {GOAL} to {START} and reverses them. "
                f"Highlighting edge {u} → {n} (cost {g[n] - g[u]}).")
    elif kind == "done":
        path = state["path"]
        costs = " + ".join(str(g[b] - g[a]) for a, b in zip(path, path[1:]))
        title = "A* SEARCH COMPLETED —\nOPTIMAL PATH FOUND"
        lines = [f"Starting node : {START}", f"Goal node     : {GOAL}",
                 f"Optimal path  : {' → '.join(path)}", f"Total cost    : {costs} = {g[GOAL]}",
                 f"Expanded nodes: {len(state['order'])}  ({', '.join(state['order'])})",
                 "Status        : SUCCESS"]
        expl = ("Yellow = optimal path. Red nodes were explored but are not on the path; "
                "blue nodes were discovered but never needed; grey nodes were never reached.")
    else:
        title, lines, expl = "SEARCH FAILED", [], "OPEN became empty: the goal is unreachable."
    return title, lines, expl


def queue_text(state):
    """Live OPEN / CLOSED contents, read straight from the real data structures."""
    valid, stale = open_entries(state)
    open_txt = "   ".join(f"{m} (f={fv})" for fv, m in valid) or "empty"
    stale_txt = ("old heap entry, will be ignored: " + "  ".join(f"{m} (f={fv})" for fv, m in stale)
                 if stale else "")
    return open_txt, stale_txt, "  →  ".join(state["order"]) or "empty"


# ============================ 3. THE WEB PAGE (Streamlit) ============================
COL = {"start": "#22c55e", "unvisited": "#3b4a63", "open": "#3b82f6", "current": "#f97316",
       "closed": "#ef4444", "goal": "#a855f7", "path": "#facc15"}
ACCENT = {"ready": "#e2e8f0", "init": COL["start"], "select": COL["current"], "goal": COL["goal"],
          "discover": COL["open"], "update": COL["goal"], "worse": "#8b9bb4", "skip": "#8b9bb4",
          "stale": "#8b9bb4", "expanded": COL["closed"], "path": COL["path"],
          "done": COL["start"], "fail": COL["closed"]}
DURATION = {"init": 3.0, "select": 3.0, "goal": 3.5, "discover": 2.5, "update": 3.5,
            "worse": 2.0, "skip": 1.5, "stale": 2.0, "expanded": 1.8, "path": 1.3}  # seconds
CURRENT_KINDS = ("init", "select", "goal", "discover", "update", "worse", "skip")
R = 38                                          # node radius in pixels


def xy(n):
    """Convert a node's tree position into pixel coordinates on the drawing."""
    x, y = POS[n]
    return 500 + x * 118, 570 - y * 112


def node_color(n, sim):
    """Colour of node n, derived directly from the real A* data structures."""
    state, (kind, cur, _) = sim["state"], sim["ev"]
    if any(n in e for e in sim["path_edges"]):
        return COL["path"]
    if n == GOAL and n in state["g"]:
        return COL["goal"]
    if n == cur and kind in CURRENT_KINDS:
        return COL["current"]
    if n == START:
        return COL["start"]
    if n in state["closed"]:
        return COL["closed"]
    return COL["open"] if n in state["g"] else COL["unvisited"]   # discovered = waiting in OPEN


def marker_route(ev, state):
    """Nodes the moving marker travels through for this event (real parent pointers)."""
    kind, n, info = ev
    if kind in ("init", "select", "goal"):
        return reconstruct_path(state["parent"], n)  # the path A* is currently extending
    if kind in ("discover", "update", "worse"):
        return [n, info[0]]
    return {"skip": [n, info], "path": [info, n]}.get(kind, [])


def apply_event(sim, ev):
    """Record one algorithm event: step counter, path edges and the dry-run table."""
    kind, n, info = ev
    if kind in ("select", "goal", "stale"):
        sim["step"] += 1
    if kind == "path":
        sim["path_edges"].add((info, n))
    if kind in ("expanded", "goal"):            # one table row per expanded node
        g = sim["state"]["g"]
        open_txt, _, closed_txt = queue_text(sim["state"])
        sim["rows"].append({"Step": sim["step"], "Expanded": n, "g(n)": g[n], "h(n)": heuristic(n),
                            "f(n)": g[n] + heuristic(n), "OPEN after": open_txt,
                            "CLOSED after": closed_txt})
    sim["ev"] = ev


SVG_DEFS = ('<defs><filter id="blur" x="-80%" y="-80%" width="260%" height="260%">'
            '<feGaussianBlur stdDeviation="9"/></filter>'
            '<filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
            + "".join(f'<marker id="arr-{k}" viewBox="0 0 10 10" refX="8" refY="5" markerUnits="userSpaceOnUse" '
                      f'markerWidth="{w}" markerHeight="{w}" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                      for k, c, w in (("dim", "#334155", 13), ("on", "#cbd5e1", 15), ("path", "#facc15", 20)))
            + '</defs>')


def render_svg(sim, speed):
    """Build the whole tree as an SVG picture (with SVG animations) from the current state."""
    state, (kind, cur, info) = sim["state"], sim["ev"]
    g, parent = state["g"], state["parent"]
    out = ['<svg viewBox="0 0 1110 640" xmlns="http://www.w3.org/2000/svg" '
           'style="width:100%;height:auto;font-family:Inter,-apple-system,Segoe UI,sans-serif">', SVG_DEFS]
    for u in GRAPH:                             # ---- edges (arrows) and their cost labels
        for v, cost in GRAPH[u]:
            (x1, y1), (x2, y2) = xy(u), xy(v)
            d = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / d, (y2 - y1) / d
            ax, ay, bx, by = x1 + ux * R, y1 + uy * R, x2 - ux * (R + 4), y2 - uy * (R + 4)
            line = f'x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}"'
            if (u, v) in sim["path_edges"]:     # optimal path: glowing yellow, drawn in
                L = math.hypot(bx - ax, by - ay)
                draw = (f'<animate attributeName="stroke-dashoffset" from="{L:.0f}" to="0" dur="0.9s" fill="freeze"/>'
                        if kind == "path" and (u, v) == (info, cur) else "")
                out.append(f'<line {line} stroke="#facc15" stroke-width="16" opacity="0.35" filter="url(#blur)"/>')
                out.append(f'<line {line} stroke="#facc15" stroke-width="6" stroke-linecap="round" '
                           f'stroke-dasharray="{L:.0f}" marker-end="url(#arr-path)">{draw}</line>')
            elif parent.get(v) == u:            # current best way to reach v
                out.append(f'<line {line} stroke="#cbd5e1" stroke-width="3" marker-end="url(#arr-on)"/>')
            else:                               # not (or no longer) used
                out.append(f'<line {line} stroke="#334155" stroke-width="2.2" marker-end="url(#arr-dim)"/>')
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            out.append(f'<rect x="{mx - 14:.0f}" y="{my - 14:.0f}" width="30" height="28" rx="8" fill="#131c31" '
                       f'stroke="#475569"/><text x="{mx:.0f}" y="{my + 5:.0f}" text-anchor="middle" '
                       f'fill="white" font-size="17" font-weight="700">{cost}</text>')
    new = info[0] if kind in ("discover", "update") else (cur if kind == "path" else None)
    for n in POS:                               # ---- glowing nodes with labels
        x, y = xy(n)
        c = node_color(n, sim)
        if c != COL["unvisited"]:
            out.append(f'<circle cx="{x}" cy="{y}" r="54" fill="{c}" opacity="0.35" filter="url(#blur)"/>')
        pop = '<animate attributeName="r" values="18;42;34" dur="0.6s"/>' if n == new else ""
        stroke = "#64748b" if c == COL["unvisited"] else "#f8fafc"
        txt = "#1c1917" if c == COL["path"] else "white"
        out.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="{c}" stroke="{stroke}" stroke-width="3">{pop}</circle>'
                   f'<text x="{x}" y="{y + 5}" text-anchor="middle" fill="{txt}" font-size="30" font-weight="800">{n}</text>'
                   f'<text x="{x}" y="{y + 25}" text-anchor="middle" fill="{txt}" font-size="14" font-weight="600" opacity="0.9">h = {heuristic(n)}</text>')
        if n in g:                              # g / f badge beside the node
            out.append(f'<rect x="{x + R + 8}" y="{y - 17}" width="132" height="34" rx="10" fill="#131c31" '
                       f'stroke="{c}" stroke-width="2"/><text x="{x + R + 74}" y="{y + 6}" text-anchor="middle" '
                       f'fill="#e2e8f0" font-size="16" font-weight="700" style="white-space:pre">g = {g[n]}   f = {g[n] + heuristic(n)}</text>')
    if kind in CURRENT_KINDS and cur:           # ---- pulsing ring + ripple on the current node
        x, y = xy(cur)
        c = COL["goal"] if cur == GOAL else COL["current"]
        out.append(f'<circle cx="{x}" cy="{y}" r="46" fill="none" stroke="{c}" stroke-width="4">'
                   f'<animate attributeName="r" values="45;50;45" dur="1s" repeatCount="indefinite"/></circle>'
                   f'<circle cx="{x}" cy="{y}" r="40" fill="none" stroke="{c}" stroke-width="3">'
                   f'<animate attributeName="r" values="44;88" dur="1.4s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.9;0" dur="1.4s" repeatCount="indefinite"/></circle>')
    route = marker_route(sim["ev"], state)
    if len(route) >= 2:                         # ---- glowing comet marker gliding along the arrows
        path = "M" + " L".join(f"{xy(n)[0]:.0f},{xy(n)[1]:.0f}" for n in route)
        dur = 0.6 * DURATION[kind] / speed
        color = {"discover": COL["open"], "update": COL["goal"], "path": COL["path"],
                 "worse": "#8b9bb4", "skip": "#8b9bb4"}.get(kind, COL["current"])
        for size, alpha, delay in ((11, 1, 0), (8, 0.55, 0.07), (6, 0.35, 0.14), (4, 0.2, 0.21)):
            out.append(f'<circle r="{size}" fill="{color}" stroke="white" stroke-width="{2 if delay == 0 else 0}" '
                       f'opacity="0" filter="url(#glow)"><set attributeName="opacity" to="{alpha}" begin="{delay}s"/>'
                       f'<animateMotion path="{path}" begin="{delay}s" dur="{dur:.2f}s" fill="freeze" calcMode="spline" '
                       f'keyPoints="0;1" keyTimes="0;1" keySplines="0.45 0 0.55 1"/>'
                       f'<animate attributeName="opacity" to="0" begin="{delay + dur:.2f}s" dur="0.3s" fill="freeze"/></circle>')
    out.append("</svg>")
    return "".join(out)


def chips(items, css):
    return "".join(f'<span class="chip {css}">{html.escape(t)}</span>' for t in items) or '<span class="none">empty</span>'


def render_panel(sim):
    """Build the explanation panel (HTML) for the current event."""
    state, ev = sim["state"], sim["ev"]
    kind, cur, info = ev
    title, lines, expl = describe(ev, state, sim["step"])
    g = state["g"]
    focus = info[0] if kind in ("discover", "update") else cur if kind in ("init", "select", "goal") else None
    metrics = ""
    if focus in g:                              # big g / h / f cards for the node in focus
        metrics = '<div class="metrics">' + "".join(
            f'<div class="metric"><div class="k">{k}({focus})</div><div class="v">{v}</div></div>'
            for k, v in (("g", g[focus]), ("h", heuristic(focus)), ("f", g[focus] + heuristic(focus)))) + "</div>"
    valid, stale = open_entries(state)
    calc = "<br>".join(html.escape(l) for l in lines)
    sec = "RESULT" if kind == "done" else "THE FORMULA" if kind == "ready" else "CALCULATION"
    return (f'<div class="panel" style="--accent:{ACCENT[kind]}">'
            f'<div class="ptitle">{html.escape(title).replace(chr(10), "<br>")}</div>{metrics}'
            f'<div class="sec">{sec}</div><div class="calc">{calc}</div>'
            f'<div class="sec">OPEN LIST · priority queue (lowest f first)</div>'
            f'<div>{chips([f"{n}  f={f}" for f, n in valid], "open")}'
            f'{chips([f"{n}  f={f} (old)" for f, n in stale], "stale") if stale else ""}</div>'
            f'<div class="sec">CLOSED SET · already expanded</div><div>{chips(state["order"], "closed")}</div>'
            f'<div class="sec">WHY THIS STEP?</div><div class="expl">{html.escape(expl)}</div></div>')


def render_header(sim):
    kind, g = sim["ev"][0], sim["state"]["g"]
    status = {"ready": "READY · PRESS START", "done": f"OPTIMAL PATH FOUND · COST {g.get(GOAL)}",
              "path": "RECONSTRUCTING PATH"}.get(kind, f"SEARCHING · STEP {sim['step']}")
    return (f'<div class="hdr"><div><div class="logo">A* Search Visualizer</div>'
            f'<div class="sub">Informed search &nbsp;·&nbsp; f(n) = g(n) + h(n) &nbsp;·&nbsp; AI &amp; Soft Computing</div></div>'
            f'<div class="status" style="--accent:{ACCENT[kind]}">{status}</div></div>')


LEGEND = '<div class="legend">' + "".join(
    f'<span><i class="dot" style="background:{c};color:{c}"></i>{t}</span>' for c, t in zip(COL.values(), (
        "Start", "Unvisited", "Discovered (OPEN)", "Current (expanding)", "Explored (CLOSED)", "Goal",
        "Optimal path"))) + '</div><div class="hint">Inside circle: node name and h(n) &nbsp;·&nbsp; badge: g(n), f(n) &nbsp;·&nbsp; number on arrow: step cost</div>'

CSS = """<style>
.stApp {background: radial-gradient(1200px 600px at 15% -10%, #1a2747 0%, #0b1120 60%) fixed;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.6rem; max-width: 1560px;}
.hdr {display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;}
.logo {font-size:2.3rem; font-weight:800; letter-spacing:-0.02em;
       background:linear-gradient(90deg,#f8fafc,#facc15 60%,#f97316); -webkit-background-clip:text; color:transparent;}
.sub {color:#8b9bb4; font-size:1rem; margin-top:2px;}
.status {border:2px solid var(--accent); color:var(--accent); padding:9px 18px; border-radius:999px;
         font-weight:800; letter-spacing:.04em; box-shadow:0 0 18px color-mix(in srgb, var(--accent) 35%, transparent);}
.treebox {background:rgba(19,28,49,.55); border:1px solid #27344f; border-radius:20px; padding:8px 12px;
          box-shadow:0 12px 40px rgba(0,0,0,.35);}
.panel {background:#131c31; border:1px solid #27344f; border-top:5px solid var(--accent); border-radius:20px;
        padding:20px 24px; color:#e2e8f0; box-shadow:0 12px 40px rgba(0,0,0,.35); min-height:600px;}
.ptitle {color:var(--accent); font-weight:800; font-size:1.2rem; letter-spacing:.02em; line-height:1.35;}
.metrics {display:flex; gap:10px; margin:14px 0 4px;}
.metric {flex:1; background:#0b1120; border:1px solid #27344f; border-radius:14px; padding:8px 14px;}
.metric .k {color:#8b9bb4; font-size:.8rem; font-weight:700;}
.metric .v {font-size:1.7rem; font-weight:800; color:var(--accent);}
.sec {color:#8b9bb4; font-size:.72rem; font-weight:800; letter-spacing:.1em; margin:16px 0 7px; text-transform:uppercase;}
.calc {font-family:"JetBrains Mono",Menlo,Consolas,monospace; background:#0b1120; border-radius:12px;
       padding:10px 14px; font-size:.9rem; line-height:1.7; white-space:pre-wrap;}
.chip {display:inline-block; padding:4px 12px; border-radius:999px; margin:3px 6px 3px 0; font-size:.88rem; font-weight:700;}
.chip.open {background:rgba(59,130,246,.18); color:#93c5fd; border:1px solid #3b82f6;}
.chip.closed {background:rgba(239,68,68,.15); color:#fca5a5; border:1px solid #ef4444;}
.chip.stale {color:#8b9bb4; text-decoration:line-through; border:1px dashed #475569;}
.none {color:#64748b; font-style:italic;}
.expl {font-size:1.02rem; line-height:1.65; background:#0b1120; border-left:4px solid var(--accent);
       border-radius:10px; padding:12px 14px;}
.legend {display:flex; flex-wrap:wrap; gap:8px 22px; justify-content:center; color:#cbd5e1; font-size:.92rem; margin-top:12px;}
.dot {width:14px; height:14px; border-radius:50%; display:inline-block; margin-right:7px; vertical-align:-2px;
      box-shadow:0 0 10px currentColor;}
.hint {text-align:center; color:#64748b; font-size:.85rem; margin-top:6px;}
.stButton > button {border-radius:12px; font-weight:800; height:2.9rem; border:1px solid #27344f;}
label, [data-testid="stWidgetLabel"] p {color:#cbd5e1 !important;}
</style>"""


def main():
    st.set_page_config(page_title="A* Search Visualizer", page_icon="⭐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state                       # remembers values between button clicks
    ss.setdefault("mode", "ready")              # ready | playing | paused | done
    ss.setdefault("index", 0)                   # how many algorithm events have been shown
    ss.setdefault("celebrated", False)

    def on_start():
        ss.mode, ss.index, ss.celebrated = "playing", 0, False

    def on_pause():
        ss.mode = "paused" if ss.mode == "playing" else "playing"

    def on_reset():
        ss.mode, ss.index = "ready", 0

    header_ph = st.empty()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3], vertical_alignment="bottom")
    c1.button("▶  Start", type="primary", on_click=on_start, width="stretch")
    c2.button("⏸  Pause" if ss.mode != "paused" else "▶  Resume", on_click=on_pause, width="stretch",
              disabled=ss.mode not in ("playing", "paused"))
    c3.button("↺  Reset", on_click=on_reset, width="stretch")
    speed = c4.slider("Animation speed", 0.25, 3.0, 1.0, 0.25, format="%.2fx")
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        tree_ph = st.empty()
        st.markdown(LEGEND, unsafe_allow_html=True)
    with right:
        panel_ph = st.empty()
        timer_ph = st.empty()
    with st.expander("📋  Dry-run table: one row per expanded node (fills in live)"):
        table_ph = st.empty()
    with st.expander("🐍  The Python A* code that is running"):
        st.code(inspect.getsource(astar_steps), language="python")

    sim = {"state": astar_search(START, GOAL), "ev": ("ready", None, None), "step": 0,
           "path_edges": set(), "rows": []}
    for _ in range(ss.index):                   # replay to where we were (after Pause / speed change)
        apply_event(sim, next(sim["state"]["steps"]))

    def draw():
        header_ph.markdown(render_header(sim), unsafe_allow_html=True)
        tree_ph.markdown(f'<div class="treebox">{render_svg(sim, speed)}</div>', unsafe_allow_html=True)
        panel_ph.markdown(render_panel(sim), unsafe_allow_html=True)
        table_ph.dataframe(sim["rows"], hide_index=True, width="stretch")

    draw()
    while ss.mode == "playing":                 # automatic playback: one real A* step at a time
        ev = next(sim["state"]["steps"])
        apply_event(sim, ev)
        ss.index += 1
        draw()
        if ev[0] in ("done", "fail"):
            ss.mode = "done"
            if not ss.celebrated:
                ss.celebrated = True
                st.balloons()
            break
        wait = DURATION[ev[0]] / speed
        for i in range(20):                     # small sleeps so Pause reacts quickly
            timer_ph.progress((i + 1) / 20, text="Next step…")
            time.sleep(wait / 20)
    timer_ph.empty()


if st.runtime.exists():                         # started by Streamlit -> build the page
    main()
elif __name__ == "__main__":                    # started with "python3 astar_web_app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__,
                              "--theme.base=dark", "--theme.primaryColor=#f97316",
                              "--theme.backgroundColor=#0b1120", "--theme.secondaryBackgroundColor=#131c31",
                              "--theme.textColor=#e2e8f0", "--browser.gatherUsageStats=false",
                              "--client.toolbarMode=viewer"]))
