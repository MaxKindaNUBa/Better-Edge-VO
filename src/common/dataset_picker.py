"""GUI for choosing which RGB-D sequence(s) to run.

Finds every sequence under <project>/datasets (any folder holding rgb.txt + depth.txt,
i.e. TUM or Bonn layout) and shows them in a Tk window. Used by demo_edges.py (one
sequence) and by run_dtvo.py / run_detection.py (one or more) whenever no sequence is
given on the command line.

    from src.common.dataset_picker import pick_sequence, pick_sequences
    seq_dir = pick_sequence("Choose a sequence")     # Path, or None if cancelled
    seq_dirs = pick_sequences("Choose sequences")    # list of Paths, empty if cancelled

In the multi-select window a click ticks / unticks a sequence, a click on TUM / BONN
ticks the whole dataset, and Run starts on every ticked sequence.

Can also be run on its own to just print the choice:
    conda run -n 312ml python -m src.common.dataset_picker
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
_LAST_CHOICE = PROJECT_ROOT / "output_videos" / ".last_sequence.json"
_ON, _OFF = "☑", "☐"


@dataclass
class SequenceInfo:
    path: Path
    dataset: str          # "tum", "bonn", ... (parent folder name)
    name: str
    frames: int           # colour frames listed in rgb.txt
    duration_s: float
    has_groundtruth: bool


def _count_list(path: Path) -> tuple[int, float]:
    stamps = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stamps.append(float(line.split()[0]))
    return len(stamps), (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0


def find_sequences(root: Path = DATASETS_DIR) -> list[SequenceInfo]:
    seqs = []
    for rgb_txt in sorted(root.rglob("rgb.txt")):
        d = rgb_txt.parent
        if not (d / "depth.txt").exists():
            continue
        n, dur = _count_list(rgb_txt)
        seqs.append(SequenceInfo(d, d.relative_to(root).parts[0] if d.parent != root else "-",
                                 d.name, n, dur, (d / "groundtruth.txt").exists()))
    return seqs


def _load_last() -> list[str]:
    try:
        last = json.loads(_LAST_CHOICE.read_text())
        return last.get("paths") or [last["path"]]
    except Exception:
        return []


def _save_last(paths: list[Path]):
    try:
        _LAST_CHOICE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_CHOICE.write_text(json.dumps({"path": str(paths[0]), "paths": [str(p) for p in paths]}))
    except Exception:
        pass


def _selector(title: str, root: Path, multiple: bool) -> list[Path]:
    import tkinter as tk
    from tkinter import ttk

    seqs = find_sequences(root)
    if not seqs:
        raise FileNotFoundError(f"No sequences (folders with rgb.txt + depth.txt) under {root}")

    win = tk.Tk()
    win.title(title)
    win.geometry("820x560")
    win.minsize(600, 300)
    style = ttk.Style(win)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Treeview", rowheight=24)
    chosen: list[Path] = []

    hint = ("click to tick one or more sequences, click TUM / BONN to tick a whole dataset"
            if multiple else "double-click or press Enter to run")
    ttk.Label(win, text=title, font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
    ttk.Label(win, text=f"{len(seqs)} sequences found in {root}   ·   {hint}",
              foreground="#555").pack(anchor="w", padx=12, pady=(0, 8))

    filt = tk.StringVar()
    fr = ttk.Frame(win)
    fr.pack(fill="x", padx=12)
    ttk.Label(fr, text="Filter:").pack(side="left")
    entry = ttk.Entry(fr, textvariable=filt)
    entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

    cols = ("frames", "duration", "gt")
    box = ttk.Frame(win)
    box.pack(fill="both", expand=True, padx=12, pady=8)
    tree = ttk.Treeview(box, columns=cols, selectmode="none" if multiple else "browse")
    tree.heading("#0", text="Sequence")
    tree.heading("frames", text="Frames")
    tree.heading("duration", text="Duration")
    tree.heading("gt", text="Ground truth")
    tree.column("#0", width=430)
    for c, w in zip(cols, (90, 100, 110)):
        tree.column(c, width=w, anchor="center")
    sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    item_seq: dict[str, SequenceInfo] = {}
    group_items: dict[str, list[str]] = {}
    group_name: dict[str, str] = {}
    last = _load_last()
    ticked: set[Path] = {s.path for s in seqs if str(s.path) in last} if multiple else set()
    count = tk.StringVar()

    def refresh_marks():
        for iid, s in item_seq.items():
            tree.item(iid, text=f"{_ON if s.path in ticked else _OFF}  {s.name}")
        for g, iids in group_items.items():
            on = sum(item_seq[i].path in ticked for i in iids)
            mark = _ON if iids and on == len(iids) else _OFF
            tree.item(g, text=f"{mark}  {group_name[g]}  ({on}/{len(iids)})")
        n = len(ticked)
        frames = sum(s.frames for s in seqs if s.path in ticked)
        count.set(f"{n} selected   ·   {frames} frames" if n else "nothing selected")

    def populate():
        tree.delete(*tree.get_children())
        item_seq.clear()
        group_items.clear()
        group_name.clear()
        q = filt.get().lower().strip()
        groups: dict[str, str] = {}
        select = None
        for s in seqs:
            if q and q not in s.name.lower() and q not in s.dataset.lower():
                continue
            if s.dataset not in groups:
                groups[s.dataset] = tree.insert("", "end", text=s.dataset.upper(), open=True)
                group_name[groups[s.dataset]] = s.dataset.upper()
                group_items[groups[s.dataset]] = []
            iid = tree.insert(groups[s.dataset], "end", text=s.name,
                              values=(s.frames, f"{s.duration_s:.0f} s", "yes" if s.has_groundtruth else "no"))
            item_seq[iid] = s
            group_items[groups[s.dataset]].append(iid)
            if select is None or str(s.path) in last[:1]:  # first row, or the last-used sequence
                select = iid
        if multiple:
            refresh_marks()
        elif select:
            tree.selection_set(select)
            tree.see(select)

    def toggle(iid: str):
        members = group_items.get(iid, [iid])
        paths = {item_seq[i].path for i in members if i in item_seq}
        if paths <= ticked:
            ticked.difference_update(paths)
        else:
            ticked.update(paths)
        refresh_marks()

    def on_click(event):
        if "indicator" in tree.identify_element(event.x, event.y):
            return None             # the expand / collapse arrow keeps working
        iid = tree.identify_row(event.y)
        if iid:
            tree.focus(iid)
            toggle(iid)
        return "break"

    def set_all(on: bool):
        visible = {s.path for s in item_seq.values()}
        if on:
            ticked.update(visible)
        else:
            ticked.difference_update(visible)
        refresh_marks()

    def accept(_event=None):
        if multiple:
            chosen.extend(s.path for s in seqs if s.path in ticked)   # in list order
        else:
            sel = tree.selection()
            if sel and sel[0] in item_seq:
                chosen.append(item_seq[sel[0]].path)
        if chosen:
            win.destroy()

    filt.trace_add("write", lambda *_: populate())
    if multiple:
        tree.bind("<Button-1>", on_click)
        tree.bind("<space>", lambda _e: tree.focus() and toggle(tree.focus()))
    else:
        tree.bind("<Double-1>", accept)
    win.bind("<Return>", accept)
    win.bind("<Escape>", lambda _e: win.destroy())

    btns = ttk.Frame(win)
    btns.pack(fill="x", padx=12, pady=(0, 12))
    if multiple:
        ttk.Button(btns, text="Select all shown", command=lambda: set_all(True)).pack(side="left")
        ttk.Button(btns, text="Clear shown", command=lambda: set_all(False)).pack(side="left", padx=6)
        ttk.Label(btns, textvariable=count, foreground="#555").pack(side="left", padx=6)
    ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
    ttk.Button(btns, text="Run", command=accept).pack(side="right", padx=6)

    populate()
    entry.focus_set()
    win.mainloop()

    if chosen:
        _save_last(chosen)
    return chosen


def pick_sequence(title: str = "Select an RGB-D sequence", root: Path = DATASETS_DIR) -> Path | None:
    """Open the selector window and block until the user chooses (None = cancelled)."""
    chosen = _selector(title, root, multiple=False)
    return chosen[0] if chosen else None


def pick_sequences(title: str = "Select RGB-D sequences", root: Path = DATASETS_DIR) -> list[Path]:
    """Multi-select version: every ticked sequence, in list order (empty = cancelled)."""
    return _selector(title, root, multiple=True)


def sequence_from_args_or_gui(arg: Path | None, title: str) -> Path:
    """Command-line sequence if given, otherwise the selector window. Exits on cancel."""
    if arg is not None:
        return arg
    try:
        seq = pick_sequence(title)
    except Exception as e:  # no display / no Tk
        sys.exit(f"Could not open the dataset selector ({e}); pass the sequence path as an argument.")
    if seq is None:
        sys.exit("No sequence selected.")
    return seq


def sequences_from_args_or_gui(args: list[Path], title: str) -> list[Path]:
    """Command-line sequences if given, otherwise the multi-select window. Exits on cancel."""
    if args:
        return list(args)
    try:
        seqs = pick_sequences(title)
    except Exception as e:  # no display / no Tk
        sys.exit(f"Could not open the dataset selector ({e}); pass the sequence path(s) as arguments.")
    if not seqs:
        sys.exit("No sequence selected.")
    return seqs


if __name__ == "__main__":
    print(pick_sequences())
