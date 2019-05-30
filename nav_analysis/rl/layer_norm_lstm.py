import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = self.ln_ih(
            F.linear(input, self.weight_ih, self.bias_ih)
        ) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.ln_ho(cy))

        return hy, (hy, cy)


class LayerNormLSTM(jit.ScriptModule):
    __constants__ = ["cells"]

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.cells = nn.ModuleList(
            [
                LayerNormLSTMCell(
                    input_size if i == 0 else hidden_size, hidden_size
                )
                for i in range(num_layers)
            ]
        )

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        hx, cx = state
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out = inputs[i]
            for j in range(len(self.cells)):
                out, (hx[j], cx[j]) = self.cells[j](out, (hx[j], cx[j]))

            outputs += [out]
        return torch.stack(outputs), (hx, cx)
