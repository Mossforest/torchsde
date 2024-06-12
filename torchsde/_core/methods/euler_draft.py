# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import warnings

from .. import base_solver
from .. import adaptive_stepping
from .. import interp
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS


class EulerDraft(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.ito
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        super(EulerDraft, self).__init__(sde=sde, **kwargs)
        self.script = kwargs.get('script', None)

    def step(self, t0, t1, y0, extra0):
        del extra0
        dt = t1 - t0
        I_k = self.bm(t0, t1)  # TODO: ?

        f, h, g_prod = self.sde.f(t0, y0), self.sde.h(t0, y0), self.sde.g(t0, y0)

        # TODO
        y1 = y0 + (f + h) * dt + g_prod
        return y1, ()

    def integrate(self, y0, ts, extra0):
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if self.adaptive:
                    raise NotImplementedError
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                    curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra