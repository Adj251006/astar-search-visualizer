# A* Search Algorithm — Interactive Tree Visualizer

An educational web app, written entirely in **Python** with **Streamlit**, that shows how the **A\* informed search algorithm** works on a search tree. Press **Start** once and the real algorithm plays automatically, step by step.

**AI and Soft Computing (AISE/AICP) academic project.**

## Features

- Real A\* implementation: `heapq` priority queue (OPEN), closed set, `g` scores and parent pointers, with **f(n) = g(n) + h(n)**
- Animated tree: nodes change colour, a marker moves along the edges, and the optimal path is highlighted at the end
- Live panel with the g / h / f calculation, the OPEN and CLOSED lists, and a plain-English explanation of every step
- Better-path update and outdated heap entries shown visually
- Dry-run table that fills in live, plus a view of the Python A\* code that is running
- Start, Pause/Resume, Reset and a speed slider

## Colour legend

| Colour | Meaning |
|---|---|
| Green | Start node |
| Grey | Unvisited |
| Blue | Discovered (in OPEN) |
| Orange | Current node being expanded |
| Red | Explored (in CLOSED) |
| Purple | Goal |
| Yellow | Optimal path |

## Sample problem

There are two routes to the goal G:

- S → A → D → G, cost **9** (the optimal route)
- S → B → E → G, cost 10

The heuristic is admissible and consistent.

## Run locally

```bash
pip install -r requirements.txt
streamlit run astar_web_app.py
```

## Files

- `astar_web_app.py`: the complete app (algorithm, drawing and web page)
- `VIVA_GUIDE.md`: code explanation, pseudocode, dry run, viva questions and complexity
- `.streamlit/config.toml`: dark theme settings
