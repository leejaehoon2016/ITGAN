import six
import math
from .layers.wrappers import cnf_regularization as reg_lib
from .layers.cnf import CNF
from .layers.odefunc import ODEfunc, ODEnet
from .layers.normalization import MovingBatchNorm1d
from .layers.container import SequentialFlow

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)



REGULARIZATION_FNS = {
    "l1int": reg_lib.l1_regularzation_fn,
    "l2int": reg_lib.l2_regularzation_fn,
    "dl2int": reg_lib.directional_l2_regularization_fn,
    "JFrobint": reg_lib.jacobian_frobenius_regularization_fn1,
    "JdiagFrobint": reg_lib.jacobian_diag_frobenius_regularization_fn,
    "JoffdiagFrobint": reg_lib.jacobian_offdiag_frobenius_regularization_fn,
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn2,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs

def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, CNF):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return acc_reg_states
def build_model_tabular(args, dims, regularization_fns=None):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            mode = args.mode,
            adjoint = args.adjoint,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)

    set_cnf_options(args, model)

    return model
