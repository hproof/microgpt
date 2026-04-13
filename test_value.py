from my_microgpt import Value


def test_value():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a
    d.backward()
    print(f"a: data={a.data}, grad={a.grad}")
    print(f"b: data={b.data}, grad={b.grad}")


if __name__ == "__main__":
    test_value()
