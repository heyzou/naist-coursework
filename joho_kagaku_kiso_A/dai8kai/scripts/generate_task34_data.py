import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)

A1 = ((0.0, 1.0), (-1.0, -3.0))
A2 = ((0.0, 1.0), (-1000.0, -1001.0))

H1 = [0.8, 0.5, 0.2, 0.1]
H2 = [0.08, 0.05, 0.002, 0.001]
T1 = 10.0
T2 = 0.5


def exact1(t: float) -> float:
    c1 = (5 + 3 * math.sqrt(5)) / 10
    c2 = (5 - 3 * math.sqrt(5)) / 10
    l1 = (-3 + math.sqrt(5)) / 2
    l2 = (-3 - math.sqrt(5)) / 2
    return c1 * math.exp(l1 * t) + c2 * math.exp(l2 * t)


def exact2(t: float) -> float:
    return (1000 / 999) * math.exp(-t) - (1 / 999) * math.exp(-1000 * t)


def euler(A, h: float, T: float):
    n = int(round(T / h))
    y0, y1 = 1.0, 0.0
    ts = [0.0]
    xs = [1.0]
    for k in range(n):
        ny0 = y0 + h * (A[0][0] * y0 + A[0][1] * y1)
        ny1 = y1 + h * (A[1][0] * y0 + A[1][1] * y1)
        y0, y1 = ny0, ny1
        ts.append((k + 1) * h)
        xs.append(y0)
    return ts, xs


def write_series(path: Path, ts, ys):
    with path.open('w', encoding='utf-8') as f:
        f.write('t x\n')
        for t, y in zip(ts, ys):
            f.write(f'{t:.12g} {y:.12g}\n')


# exact curves
n1 = 1000
te1 = [T1 * i / n1 for i in range(n1 + 1)]
xe1 = [exact1(t) for t in te1]
write_series(OUT / 'task34_sys1_exact.dat', te1, xe1)

n2 = 1000
te2 = [T2 * i / n2 for i in range(n2 + 1)]
xe2 = [exact2(t) for t in te2]
write_series(OUT / 'task34_sys2_exact.dat', te2, xe2)

# euler curves
for h in H1:
    t, x = euler(A1, h, T1)
    tag = str(h).replace('.', 'p')
    write_series(OUT / f'task34_sys1_h_{tag}.dat', t, x)

for h in H2:
    t, x = euler(A2, h, T2)
    tag = str(h).replace('.', 'p')
    write_series(OUT / f'task34_sys2_h_{tag}.dat', t, x)

print('generated')
