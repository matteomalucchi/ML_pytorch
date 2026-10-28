"""ONNX helpers shared by ml_onnx (convert_to_onnx.py) and the k-fold merge
(kfold_merge.py): saving/running models and building, with spox, the graphs
of the ratio of the outputs and of the combination of several models."""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import spox.opset.ai.onnx.v17 as op
from spox import Tensor, argument, build, inline

# Numerical guards for the multiclass morphing-weight ratio (see
# get_multiclass_ratio_model_tensor_onnx). The denominator is a relu, so it is
# exactly 0 for every event the classifier considers more ttbar- than QCD-like
# in the denominator region; relu(num) / 0 is +inf (or 0/0 -> nan). Because
# ml_onnx averages the per-model weights *after* the division, a single sub-model
# hitting 0 poisons the ensemble weight for that event. RATIO_DEN_EPS floors the
# denominator so no inf/nan is produced; RATIO_W_MAX bounds the resulting
# per-model weight so a near-zero denominator gives a large-but-finite value
# instead of a spike that dominates avg_w and any downstream histogram bin.
# Lower RATIO_W_MAX if a few high-weight events still dominate downstream.
RATIO_DEN_EPS = 1e-6


def save_onnx_model(onnx_model_final, onnx_model_name):
    if os.path.exists(onnx_model_name):
        print(f"Removing {onnx_model_name}")
        os.remove(onnx_model_name)
    onnx.save(onnx_model_final, onnx_model_name)
    print(f"Model saved as {onnx_model_name}")


def get_onnx_output(onnx_model_name, input_data):
    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        onnx_model_name, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # print the input/output name and shape
    input_name = [input.name for input in session.get_inputs()]
    output_name = [output.name for output in session.get_outputs()]

    input_shape = [input.shape for input in session.get_inputs()]
    output_shape = [output.shape for output in session.get_outputs()]

    input_example = {input_name[0]: input_data}
    output_onnx = session.run(output_name, input_example)

    return output_onnx



def get_ratio_model_tensor_onnx(onnx_model, b):
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    # get the output shape of the model
    output_shape = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    )
    print(f"Output shape: {output_shape}")

    # To take the ratio of the first model too.
    (r,) = inline(onnx_model)(b).values()
    print(f"{b.type = !s}, {r.type = !s}")
    if output_shape == 1:
        r = op.div(r, op.sub(op.const(1.0, dtype="float32"), r))
    elif output_shape == 2:
        r_0 = op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, 0])),
                op.constant(value=np.array([sys.maxsize, 1])),
            ),
            axes=op.const([-1]),
        )
        r_1 = op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, 1])),
                op.constant(value=np.array([sys.maxsize, 2])),
            ),
            axes=op.const([-1]),
        )
        r = op.div(r_1, r_0)
        # r = op.div(r_0, r_1)
        print(f"{r_0.type = !s}, {r_1.type = !s}, {r.type = !s}")
    else:
        raise ValueError("The output shape is not 1 or 2")

    return r

def get_multiclass_num_den_tensor_onnx(onnx_model, b, alpha, alpha_den):
    """Build the *raw*, un-clamped numerator/denominator tensors for one
    k-fold multiclass model -- i.e. ``r_0 - alpha*r_2`` and
    ``r_1 - alpha_den*r_3`` with no ``relu`` and no epsilon floor applied.

    This exists so the per-fold numerator and denominator can be summed
    across all k-fold models and averaged *before* non-negativity is
    enforced (see ``main()``), instead of enforcing it on each fold
    individually and then averaging the already-clipped per-fold ratios
    (what ``get_multiclass_ratio_model_tensor_onnx`` does, still used for
    the per-fold diagnostic outputs in ``all_ratios_model_*.onnx``).

    Clipping every fold separately before averaging is worse than clipping
    the average once: (1) ``relu`` is convex, so by Jensen's inequality
    ``E[relu(x)] >= relu(E[x])`` -- averaging noisy per-fold values *after*
    the floor systematically overestimates the true (non-negative)
    quantity whenever it sits near zero; (2) whenever a single fold's
    estimate dips negative for a given event, it gets hard-zeroed for that
    fold regardless of what the other folds say, instead of letting the
    folds' independent statistical noise partially cancel in the average
    first. Averaging the raw numerator/denominator first and clipping once
    avoids both effects.
    """
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    output_shape = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    )
    if output_shape != 4:
        raise ValueError(
            f"Multiclass morphing expects a 4-node model, got {output_shape} outputs"
        )

    (r,) = inline(onnx_model)(b).values()

    def _node(i):
        return op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, i])),
                op.constant(value=np.array([sys.maxsize, i + 1])),
            ),
            axes=op.const([-1]),
        )

    r_0 = _node(0)  # data,  numerator region
    r_1 = _node(1)  # data,  denominator region
    r_2 = _node(2)  # ttbar, numerator region
    r_3 = _node(3)  # ttbar, denominator region

    if alpha:
        numerator = op.sub(r_0, op.mul(op.const(float(alpha), dtype="float32"), r_2))
    else:
        numerator = r_0
    if alpha_den:
        denominator = op.sub(r_1, op.mul(op.const(float(alpha_den), dtype="float32"), r_3))
    else:
        denominator = r_1

    return numerator, denominator


def get_multiclass_ratio_model_tensor_onnx(onnx_model, b, alpha, alpha_den):
    """Build the QCD-morphing weight tensor for a 4-node multiclass model.

    Output-node convention (class_idx order, i.e. sorted by ``lbl``)::

        r_0 = p(data,  numerator region)    e.g. boosted_signal_sideband_region_C
        r_1 = p(data,  denominator region)  e.g. boosted_control_sideband_region_A
        r_2 = p(ttbar, numerator region)
        r_3 = p(ttbar, denominator region)

    The weight reweights data in the denominator region so that it reproduces
    the QCD shape of the numerator region::

        w(x) = clip(relu(r_0 - alpha * r_2) / (relu(r_1 - alpha_den * r_3) + eps),
                    0, RATIO_W_MAX)

    ``alpha = N_ttbar / N_data`` in the numerator region and ``alpha_den =
    N_ttbar / N_data`` in the denominator region. They are needed because
    training balances every class to equal total weight, so ``r_0``..``r_3``
    are all unit-normalised shapes; the alphas restore the physical ttbar
    fraction before each subtraction. ``alpha = 0`` / ``alpha_den = 0`` disable
    the corresponding subtraction (``alpha = alpha_den = 0`` gives a pure
    data-shape morph ``w = r_0 / r_1``). Only the shape of ``w`` matters; its
    overall scale is fixed downstream when the morphed sample is renormalised.

    ``alpha_den`` must match the density of the sample the weight is multiplied
    onto downstream: use ``alpha_den = 0`` when reweighting *raw* data in the
    denominator region (denominator = data-D density ``r_1``), and
    ``alpha_den = N_ttbar/N_data`` in region D when reweighting a
    ttbar-subtracted ``data_D - ttbar_D_MC`` sample (denominator = QCD-D
    density).

    ``eps`` (``RATIO_DEN_EPS``) floors the denominator: ``relu(r_1 - alpha_den *
    r_3)`` is exactly 0 for every event the classifier deems more ttbar- than
    QCD-like in the denominator region, and ``relu(num) / 0`` is ``+inf`` (or
    ``0/0 -> nan``). Since ``main()`` averages the per-model weights *after* this
    division, one sub-model hitting 0 would poison the ensemble weight for that
    event. The final ``clip`` to ``RATIO_W_MAX`` keeps a near-zero denominator
    from turning into a spike that dominates ``avg_w`` and any downstream bin.
    """
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    # get the output shape of the model
    output_shape = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    )
    print(f"Output shape: {output_shape}")
    if output_shape != 4:
        raise ValueError(
            f"Multiclass morphing expects a 4-node model, got {output_shape} outputs"
        )

    # To take the ratio of the first model too.
    (r,) = inline(onnx_model)(b).values()
    print(f"{b.type = !s}, {r.type = !s}")

    def _node(i):
        return op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, i])),
                op.constant(value=np.array([sys.maxsize, i + 1])),
            ),
            axes=op.const([-1]),
        )

    r_0 = _node(0)  # data,  numerator region
    r_1 = _node(1)  # data,  denominator region
    r_2 = _node(2)  # ttbar, numerator region
    r_3 = _node(3)  # ttbar, denominator region

    if alpha:
        # relu guards against the subtraction overshooting where the classifier
        # thinks ttbar dominates the data shape (would give a negative weight)
        numerator = op.relu(
            op.sub(r_0, op.mul(op.const(float(alpha), dtype="float32"), r_2))
        )
    else:
        numerator = r_0
    if alpha_den:
        denominator = op.relu(
            op.sub(r_1, op.mul(op.const(float(alpha_den), dtype="float32"), r_3))
        )
    else:
        denominator = r_1
    # floor the denominator so a relu that clamps to 0 cannot produce inf/nan
    denominator = op.add(denominator, op.const(float(RATIO_DEN_EPS), dtype="float32"))
    r = op.div(numerator, denominator)
    # bound the per-model weight: a near-zero denominator would otherwise give a
    # finite-but-huge spike that dominates the averaged weight downstream
    print(
        f"{r_0.type = !s}, {r_1.type = !s}, {r_2.type = !s}, {r_3.type = !s}, "
        f"{r.type = !s}"
    )
    return r


def get_score_tensor_onnx(onnx_model, b):
    """Output of a single model applied to the input tensor ``b``."""
    (r,) = inline(onnx_model)(b).values()
    return r


def _select_per_fold(tensors, event, k_folds):
    """Select, event by event, the tensor of the fold ``event % k_folds``:
    ``tensors[k]`` is used for the events with ``event % k_folds == k``."""
    fold = op.mod(event, op.const(np.int64(k_folds)))
    out = tensors[-1]
    for k in reversed(range(k_folds - 1)):
        cond = op.equal(fold, op.const(np.int64(k)))
        if len(tensors[k].type.shape) == 2:
            # broadcast the (N,) condition over the output nodes
            cond = op.unsqueeze(cond, axes=op.const([1]))
        out = op.where(cond, tensors[k], out)
    return out


def build_kfold_models(onnx_models, n_features, ratio=False, event_split=False):
    """Build the ONNX models combining the k-fold models ``onnx_models``
    (ordered by fold).

    Returns a dict {name: onnx model} with:
      - ``average``: average of the outputs of the folds (input ``args_0``)
      - ``average_ratio`` (ratio=True): average of the ratio of the outputs
        of the folds (see get_ratio_model_tensor_onnx), for reweighting
      - ``event`` (event_split=True): inputs ``args_0`` and ``event`` (int64
        event number); every event is evaluated with the model of the fold
        ``event % k_folds``, i.e. the model that was not trained on it
      - ``event_ratio`` (event_split and ratio): same with the ratio
    """
    k_folds = len(onnx_models)
    b = argument(Tensor(np.float32, ("N", n_features)))
    scores = [get_score_tensor_onnx(m, b) for m in onnx_models]
    models = {
        "average": build(
            {"args_0": b},
            {"avg_score": op.div(_sum(scores), op.const(np.float32(k_folds)))},
        )
    }
    if ratio:
        ratios = [get_ratio_model_tensor_onnx(m, b) for m in onnx_models]
        models["average_ratio"] = build(
            {"args_0": b},
            {"avg_w": op.div(_sum(ratios), op.const(np.float32(k_folds)))},
        )
    if event_split:
        event = argument(Tensor(np.int64, ("N",)))
        models["event"] = build(
            {"args_0": b, "event": event},
            {"Output": _select_per_fold(scores, event, k_folds)},
        )
        if ratio:
            models["event_ratio"] = build(
                {"args_0": b, "event": event},
                {"w": _select_per_fold(ratios, event, k_folds)},
            )
    return models


def _sum(tensors):
    s = tensors[0]
    for t in tensors[1:]:
        s = op.add(s, t)
    return s
