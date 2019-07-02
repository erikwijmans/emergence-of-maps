import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.fc_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)

        self.ln_ho = nn.LayerNorm(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = state
        ih = self.ln_ih(self.fc_ih(input))
        hh = self.ln_hh(self.fc_hh(hx))

        ingate, forgetgate, cellgate, outgate = torch.chunk(ih + hh, 4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.ln_ho(cy))

        return hy, cy


class LayerNormLSTM(jit.ScriptModule):
    __constants__ = ["cells"]

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size, hidden_size)
                if i == 0
                else LayerNormLSTMCell(hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        hx, cx = state
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out = inputs[i]
            j = 0
            for cell in self.cells:
                hx[j], cx[j] = cell(out, (hx[j], cx[j]))
                out = hx[j]
                j += 1

            outputs += [out]

        return torch.stack(outputs), (hx, cx)
