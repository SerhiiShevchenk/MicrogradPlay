from micrograd.engine import Value
from micrograd.nn import MLP
from drawer.drawer import *

input_vals = [
    [1.0, 2.0, 3.0],
    [3.0, 8.0, 2.0],
    [7.0, 8.0, 3.0],
    [3.0, 5.0, 2.0],
    [6.0, 5.0, 2.0]
]

out_vals = [6.0, 48.0, 168.0, 30.0, 60.0]
model = MLP(3, [20, 20, 10, 1])

for n in range(20):
    count = 0
    ypred = [model(x) for x in input_vals]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(out_vals, ypred))

    print('LOSS: ', loss)
    draw_dot(loss)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data += -0.1 * p.grad
