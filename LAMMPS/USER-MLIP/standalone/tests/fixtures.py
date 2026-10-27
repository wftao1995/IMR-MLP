#!/usr/bin/env python3
"""Small, independent analytic fixtures; not production benchmark potentials."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path

HEADS = ("li", "q2", "fs", "n1", "n2")
CELL = [[12.0, 1.1, -0.7], [0.0, 12.0, 0.8], [0.0, 0.0, 12.0]]
FRACTIONAL = [[.06,.12,.18], [.26,.15,.20], [.49,.17,.23], [.80,.14,.20],
              [.08,.35,.21], [.30,.37,.25], [.52,.39,.20], [.83,.36,.24]]
TYPES = [0, 1, 0, 1, 1, 0, 1, 0]

def matvec(matrix, vector):
    return [sum(a*b for a,b in zip(row, vector)) for row in matrix]

def description(head):
    q = 1 if head == "li" else (3 if head in ("n1", "n2") else 2)
    weights = {"li": [], "q2": [.7,-.2,.08,.03,-.04], "fs": [.2],
               "n1": [.07*math.cos(i+.4) for i in range(3*4+8)],
               "n2": [.07*math.cos(i+.4) for i in range(3*4+4+4*2+4)]}[head]
    return {"head": head, "q": q, "h1": 4, "h2": 2,
            "species": [.02,-.01], "bias": [[.03/(p+1),-.02/(p+1)] for p in range(q)],
            "projection": [[.04/(p+1),-.003*(p+1),.005*(p+1)] for p in range(q)],
            "weights": weights}

def write_models(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for head in HEADS:
        model = description(head)
        tag = {"schema":"gpml.mtp.metadata", "version":1, "type":"MTP_M", "model":"mtp_m_"+head, "q":model["q"],
               "model_coeffs_count":len(model["weights"]), "elements":["Al","Ni"]}
        if head in ("n1", "n2"):
            tag["nn"] = {"activation":"TANH", "hidden_layers":[4] if head == "n1" else [4,2]}
        def vector(values): return "{"+", ".join(format(x,'.17g') for x in values)+"}"
        lines = ["MTP", "version = 1.1.0", "potential_name = standalone_analytic_"+head,
                 "scaling = 1", "species_count = 2", "potential_tag = "+json.dumps(tag,separators=(',',':')),
                 "radial_basis_type = RBChebyshev", "min_dist = 0.5", "max_dist = 4.8",
                 "radial_basis_size = 2", "radial_funcs_count = 1", "radial_coeffs"]
        for center in range(2):
            for neighbor in range(2):
                lines += [f"{center}-{neighbor}", vector([.025*(1+.2*center+.1*neighbor),.01*(1+.1*center-.1*neighbor)])]
        lines += ["alpha_moments_count = 2", "alpha_index_basic_count = 1",
                  "alpha_index_basic = {{0, 0, 0, 0}}", "alpha_index_times_count = 2",
                  "alpha_index_times = {{0, 0, 1, 1}, {0, 0, 1, 1}}",
                  "alpha_scalar_moments = 3", "alpha_moment_mapping = {0, 1, 1}",
                  "species_coeffs = "+vector(model["species"]+[x for row in model["bias"] for x in row]),
                  "moment_coeffs = "+vector([x for row in model["projection"] for x in row]),
                  "model_coeffs = "+vector(model["weights"])]
        (directory/(head+".mtp")).write_text("\n".join(lines)+"\n")
    (directory/"fixture.json").write_text(json.dumps({"cell":CELL,"fractional":FRACTIONAL,"types":TYPES},indent=2)+"\n")

def head_energy(model, q):
    head, w = model["head"], model["weights"]
    if head == "li": return q[0]
    if head == "q2": return w[0]*q[0]+w[1]*q[1]+w[2]*q[0]**2+w[3]*q[1]**2+w[4]*q[0]*q[1]
    if head == "fs": return q[0]+w[0]*math.asinh(q[1])
    h1, n = model["h1"], model["q"]
    if head == "n1":
        return sum(w[n*h1+h]*math.tanh(sum(w[h*n+p]*q[p] for p in range(n))+w[n*h1+h1+h]) for h in range(h1))
    b1 = n*h1
    hidden1 = [math.tanh(sum(w[h*n+p]*q[p] for p in range(n))+w[b1+h]) for h in range(h1)]
    w2 = b1+h1
    b2 = w2+h1*model["h2"]
    return sum(w[b2+model["h2"]+h]*math.tanh(sum(w[w2+h*h1+p]*hidden1[p] for p in range(h1))+w[b2+h]) for h in range(model["h2"]))

def reference_energy(head, positions, cell=CELL, types=TYPES):
    """Independent scalar implementation; PBC sums include image multiplicity."""
    model = description(head)
    energies = []
    shifts = [matvec(cell,n) for n in itertools.product((-1,0,1),repeat=3)]
    for i, (xi, center) in enumerate(zip(positions, types)):
        m0 = 0.0
        for j, (xj, neighbor) in enumerate(zip(positions, types)):
            for shift in shifts:
                if i == j and shift == [0,0,0]: continue
                r = math.sqrt(sum((xj[k]+shift[k]-xi[k])**2 for k in range(3)))
                if r >= 4.8: continue
                t = (2*r-.5-4.8)/(4.8-.5)
                m0 += (r-4.8)**2*(.025*(1+.2*center+.1*neighbor)+.01*(1+.1*center-.1*neighbor)*t)
        basis = [m0,2*m0*m0,2*m0*m0]
        q = [model["bias"][p][center]+sum(a*b for a,b in zip(model["projection"][p],basis)) for p in range(model["q"])]
        energies.append(model["species"][center]+head_energy(model,q))
    return sum(energies), energies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    write_models(parser.parse_args().directory)
