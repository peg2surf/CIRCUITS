import matplotlib.pyplot as plt
import itertools
import numpy as np
from circuit_sim import simulate

def run():
    GRID_SIZE = 10
    wires = []
    inP = []
    sources = []
    sinks = []
    outP = []

    def or_gate(origin, wires):
        wires.append((origin[0]-1, origin[1], origin[0]-1, origin[1]+1))
        wires.append((origin[0]-1, origin[1]+1, origin[0], origin[1]+1))

        wires.append((origin[0]+1, origin[1], origin[0]+1, origin[1]+1))
        wires.append((origin[0]+1, origin[1]+1, origin[0], origin[1]+1))

        wires.append((origin[0], origin[1]+1, origin[0], origin[1]+2))

        return [(origin[0]-1, origin[1]), (origin[0]+1, origin[1])], [(origin[0], origin[1]+2)]

    def not_gate(origin: tuple, wires: list, sources: list, sinks: list, flip: bool =False):
        wires.append((*origin, origin[0], origin[1] + 3))
        if flip:
            wires.append((origin[0]-1, origin[1] + 2, origin[0]+1, origin[1] + 2))
        else:
            wires.append((origin[0]+1, origin[1] + 2, origin[0]-1, origin[1] + 2))

        sources.append(origin)
        if flip:
            sinks.append((origin[0]+1, origin[1] + 2))
            return (origin[0]-1, origin[1] + 2), (origin[0], origin[1] + 3)
        else:
            sinks.append((origin[0]-1, origin[1] + 2))
            return (origin[0]+1, origin[1] + 2), (origin[0], origin[1] + 3)

    def and_gate(origin, wires, sources, sinks, flip: bool = True):

        m = 1 if flip else -1
        nor = (origin[0], origin[1] + 3)
        output = (origin[0], origin[1] + 5)

        a, b = not_gate((origin[0]+1,origin[1]), wires, sources, sinks)
        c, d = not_gate((origin[0]-1,origin[1]), wires, sources, sinks, True)
        wires.append((*b, *nor))
        wires.append((*d, *nor))
        wires.append((*nor, *output))

        wires.append((nor[0] - 2*m, nor[1] + 1, nor[0] + 1*m, nor[1] + 1))
        sources.append((nor[0] - 2*m, nor[1] + 1))
        sinks.append((nor[0], nor[1] + 2))

        return [(a[0], origin[1]+2), (c[0], origin[1]+2)], [(nor[0] + 1*m, nor[1] + 1)]


    def xor_gate(origin, wires, sources, sinks):

        in1 = (origin[0]+2, origin[1])
        in2 = (in1[0]+1,in1[1]+1)

        i1, o1 = and_gate((in2[0]-4,in2[1]), wires=wires, sources=sources, sinks=sinks)

        wires.append((*in2, *i1[0]))
        wires.append((*in2, in2[0] -1, in2[1]))

        wires.append((*in1, i1[1][0], in1[1]))
        wires.append((i1[1][0], in1[1], *i1[1]))


        wires.append((*in1, in1[0], in1[1]+1))
        wires.append((in1[0], in1[1]+1, in1[0], in1[1]+6))

        wires.append((*o1[0], in2[0], in2[1]+4))
        sinks.append((in2[0], in2[1]+4))

        return [in1, in2], [(in1[0], in1[1]+6)]

    i1, o1 = xor_gate((6, 2), wires, sources, sinks)

    for j, i in enumerate(i1):
      inP.append((*i, True))

    for o in o1:
      outP.append(o)

    wires = np.fromiter((w for w in wires if not(w[0]==w[2] and w[1]==w[3])), dtype=np.dtype((int, 4)))
    inP = set(inP)
    sources = set(sources)
    sinks = set(sinks)

    o = np.fromiter((j for s in itertools.chain(inP, sources) for j, t in enumerate(wires) if s[0] == t[0] and s[1] == t[1] and (len(s) != 3 or s[2])), dtype=int)

    powered, _out = simulate(o, wires, sinks, outP)

    

    # Plotting the circuit and signal path
    fig, ax = plt.subplots()

    # Draw grid
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.grid(which='both')

    # Draw lines
    for i, line in enumerate(wires):
        kwargs = {}
        x1, y1, x2, y2 = line
        x = (x2 + x1)/2
        y = (y2 + y1)/2
        if powered is not None and powered[i] != 0:
            ax.plot([x1, x2], [y1, y2], 'b', label=i)
            kwargs["color"] = "blue"
        else:
            ax.plot([x1, x2], [y1, y2], 'r', label=i)
            kwargs["color"] = "red"
        ax.plot(x1, y1, 'o', **kwargs)
        ax.plot(x2, y2, 'o', **kwargs)
        ax.text(x, y, f"{i}", backgroundcolor="white", ha="center", va="center", zorder=2.5, clip_on=True, bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2'), **kwargs)

    # Mark source and sink
    for source in sources:
        ax.plot(source[0], source[1], 'o', color="green")   # Source in green
    for sink in sinks:
        ax.plot(sink[0], sink[1], 'o', color="black")       # Sink in black

    for i in inP:
        if i[2]:
            ax.plot(i[0], i[1], 'o', color="cyan")
        else:
            ax.plot(i[0], i[1], 'o', color="dimgray")

    for i, o in enumerate(outP):
        ax.plot(o[0], o[1], 'ko')
        ax.text(o[0], o[1], f"Output: {_out[i]}", ha="center", va="center", bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2'))

    # Set axis limits and labels
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title('Circuit Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

if __name__ == "__main__":
    run()