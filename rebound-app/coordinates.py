"""The canvas frame and the model frame, and why they are not the same frame.

This is the module to read first. Every other coordinate question in the app reduces
to it, and the original `webapp.py` handled it in a way that looks like a bug and is
not.

**The canvas.** `public/script.js` draws a 500x470 SVG and posts `xy[0] / 10` and
`xy[1] / 10`, so canvas coordinates are feet: ``cx`` in ``[0, 50]`` across the screen
and ``cy`` in ``[0, 47]`` down it. The court is drawn portrait -- 50 ft of court
*width* across, 47 ft of half-court *length* down.

**The model.** The pipeline's folded frame is the other way round: ``x`` in ``[0, 47]``
running from half court to the baseline, ``y`` in ``[0, 50]`` across the floor, rim at
``(41.75, 25)``.

So the two frames have their axes swapped, and the ranges prove it rather than merely
suggesting it: 500 px / 10 gives 50 ft, which can only be the 50 ft width, and 470 px
/ 10 gives 47 ft, which can only be the 47 ft half-court length. Screen across is
model ``y``; screen down is model ``x``.

**This corrects §3.4 of the handoff brief.** That section reads::

    self.pos['x']    = self.posArr[:,0]
    self.pos['y']    = self.posArr[:,1]
    self.pos['newy'] = self.posArr[:,0]     # <- newy gets column 0
    self.pos['newx'] = self.posArr[:,1]     # <- newx gets column 1

and calls it "the most damaging bug in the app" -- predicted positions drawn
transposed. It is not a bug. ``x``/``y`` are model coordinates, kept for recomputing
features; ``newx``/``newy`` are the same points converted back to the canvas, which is
exactly this swap. The brief's §7 step 3 says to "fix" it with a one-line change; doing
that would *introduce* the transposition it describes. The input side has the matching
swap, in the line ``df.columns = ['Off', 'isShoot', 'y', 'x']``.

**The length axis is confirmed.** Rohan states the canvas shows only the half of the
court containing the basket, with the basket at the *bottom* of the image. SVG ``y``
grows downward, so the bottom of the canvas is large ``cy``; the basket end of the
model frame is large ``x``, with the rim at ``x = 41.75`` of 47. Large ``cy`` therefore
means large ``x`` and no flip is needed -- ``FLIP_LENGTH`` stays ``False``. This also
agrees with the sample play, which under no flip puts a rebounder 6 ft from the rim and
the whole scene within 25 ft, where flipping strands every player 12 ft or more away.

**The width axis is settled too, and the reasoning is worth keeping.** SportVU is a
true bird's-eye view of the whole floor -- that is *why* the training data has to be
folded in the first place, since both baskets are recorded -- so the model frame carries
real-world chirality, and :func:`rebounding.data.court.fold` folds with a 180 degree
rotation, which preserves it. (A reflection would not, which is the bug the pipeline's
``abs(x - 47)`` predecessor had.)

The swap below therefore has to preserve chirality as well, and it does, for a reason
that is easy to talk yourself out of. As bare arithmetic ``(cx, cy) -> (cy, cx)`` is a
reflection: its determinant is -1. But SVG ``y`` points *down*, so the canvas is already
left-handed as drawn, and the two orientation reversals cancel. Facing the basket on the
canvas means facing screen-down, where a player's left hand points screen-right (+``cx``);
facing the basket in the model frame means facing +``x``, where his left hand points
+``y``. ``model_y = cx`` maps one to the other. Orientation is preserved and
``FLIP_WIDTH`` stays ``False``.

**And it barely matters even if this is ever got wrong.** Mirroring every shot in the
test split about the length axis and rescoring gives 30.1% top-1 either way -- identical
to a tenth of a point, against a 0.60 point standard error, with log loss 1.857 against
1.859. The same player is picked on 85.2% of shots and a player's probability moves by
0.012 on average. The model has not learned a usable left/right asymmetry, so a wrong
``FLIP_WIDTH`` costs nothing measurable. A wrong ``FLIP_LENGTH``, by contrast, would put
every player at the wrong distance from the rim.
"""

from __future__ import annotations

import numpy as np

from rebounding.constants import COURT_WIDTH, HALF_COURT_X

# Feet across and down the canvas, from the SVG's 500x470 at ten pixels per foot.
CANVAS_WIDTH_FT = COURT_WIDTH      # 50, and therefore the model's y axis
CANVAS_HEIGHT_FT = HALF_COURT_X    # 47, and therefore the model's x axis

# The basket is at the bottom of the canvas and SVG y grows downward, so canvas-down
# and model-x already agree. Confirmed by Rohan, 2026-08-19; do not change this without
# a fresh round-trip check.
FLIP_LENGTH = False

# Confirmed: SportVU is a true bird's-eye view and the fold is a rotation, so the
# training frame carries real chirality, and the swap below preserves it once SVG's
# downward y is accounted for. Mirroring the test split costs 0.0 points of top-1
# anyway. See the module docstring.
FLIP_WIDTH = False


def canvas_to_model(cx, cy) -> tuple[np.ndarray, np.ndarray]:
    """Screen feet to the pipeline's folded frame."""
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    model_x = CANVAS_HEIGHT_FT - cy if FLIP_LENGTH else cy
    model_y = CANVAS_WIDTH_FT - cx if FLIP_WIDTH else cx
    return model_x, model_y


def model_to_canvas(model_x, model_y) -> tuple[np.ndarray, np.ndarray]:
    """The inverse, for drawing a predicted position back on the court."""
    model_x = np.asarray(model_x, dtype=float)
    model_y = np.asarray(model_y, dtype=float)
    cy = CANVAS_HEIGHT_FT - model_x if FLIP_LENGTH else model_x
    cx = CANVAS_WIDTH_FT - model_y if FLIP_WIDTH else model_y
    return cx, cy
