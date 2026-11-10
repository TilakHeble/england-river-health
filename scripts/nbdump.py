"""Print text outputs of a notebook (for checking claims)."""
import sys, nbformat
for p in sys.argv[1:]:
    nb = nbformat.read(p, as_version=4); print("=====", p)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code": continue
        for o in c.get("outputs", []):
            txt = o.get("text") or o.get("data", {}).get("text/plain", "")
            if o.get("output_type") == "error": txt = "ERROR " + o["ename"] + ": " + o["evalue"]
            if txt: print(f"[{i}]", txt[:1500])
