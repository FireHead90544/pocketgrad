from engine import Tensor

if __name__ == "__main__":
    a = Tensor(0.3)
    b = Tensor(-1.7)
    c = Tensor(5.2)
    d = a * b / c
    e = d ** 2
    f = e + 1

    f.backward() # Build Computation graph and accumulate gradients via backward chaining

    print(f"Value (a): {a.data:.6f} | Gradient (df/da): {a.grad:.6f}")
    print(f"Value (b): {b.data:.6f} | Gradient (df/db): {b.grad:.6f}")
    print(f"Value (c): {c.data:.6f} | Gradient (df/dc): {c.grad:.6f}")
    print(f"Value (d): {d.data:.6f} | Gradient (df/dd): {d.grad:.6f}")
    print(f"Value (e): {e.data:.6f} | Gradient (df/de): {e.grad:.6f}")
    print(f"Value (f): {f.data:.6f} | Gradient (df/df): {f.grad:.6f}")
