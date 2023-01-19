import random
import torch
import torch.nn as nn
from .ffjord.train_misc import standard_normal_logprob, set_cnf_options, create_regularization_fns
from .ffjord.train_misc import get_regularization, build_model_tabular

class argument:
    def __init__(self, args, data_dims):
        assert args['layer_type'] in ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend", "blenddiv1", "blenddiv2", "simblenddiv1", "simblenddiv2"]
        self.layer_type = args['layer_type']
        assert type(args['hdim_factor']) is float
        self.hdim_factor = args['hdim_factor']
        assert type(args['nhidden']) is int
        self.nhidden = args['nhidden']
        assert type(args['num_blocks']) is int
        self.num_blocks = args['num_blocks']
        assert type(args['time_length']) is float
        self.time_length = args['time_length']
        assert args['train_T'] in [True, False]
        self.train_T = args['train_T']
        assert args['divergence_fn'] in ["brute_force", "approximate"]
        self.divergence_fn = args['divergence_fn']
        assert args['nonlinearity'] in ["sigmoid", "tanh", "relu", "softplus", "elu", "swish", "square", "identity", "tanh batchnorm", "relu batchnorm", "softplus batchnorm", "elu batchnorm", "swish batchnorm", "square batchnorm", "identity batchnorm"]
        self.nonlinearity = args['nonlinearity']
        assert args['test_solver'] in ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
        self.test_solver = args['test_solver']
        self.solver = args['solver']
        assert type(args['test_atol']) is float
        self.test_atol = args['test_atol']
        self.atol = args['atol']
        assert type(args['test_rtol']) is float
        self.test_rtol = args['test_rtol']
        self.rtol = args['rtol']
        assert type(args['step_size']) is float or args['step_size'] is None
        self.step_size = args['step_size']
        self.test_step_size = args['step_size']
        assert type(args['first_step']) is float or args['first_step'] is None
        self.first_step = args['first_step']   
        self.test_first_step = args['first_step']
        assert args['residual'] in [True, False]
        self.residual = args['residual']
        assert args['rademacher'] in [True, False]
        self.rademacher = args['rademacher']
        assert args['batch_norm'] in [True, False]
        self.batch_norm = args['batch_norm']
        assert type(args['bn_lag']) is float
        self.bn_lag = args['bn_lag']
        if "adjoint" in args:
            assert type(args['adjoint']) is bool
            self.adjoint = args['adjoint']
        else:
            self.adjoint = True

        assert type(args['l1int']) is float or args['l1int'] is None
        self.l1int = args['l1int']
        assert type(args['l2int']) is float or args['l2int'] is None
        self.l2int = args['l2int']
        assert type(args['dl2int']) is float or args['dl2int'] is None
        self.dl2int = args['dl2int']
        assert type(args['JFrobint']) is float or args['JFrobint'] is None
        self.JFrobint = args['JFrobint']
        assert type(args['JdiagFrobint']) is float or args['JdiagFrobint'] is None
        self.JdiagFrobint = args['JdiagFrobint']
        assert type(args['JoffdiagFrobint']) is float or args['JoffdiagFrobint'] is None
        self.JoffdiagFrobint = args['JoffdiagFrobint']


        assert type(args['kinetic_energy']) is float or args['kinetic_energy'] is None
        self.kinetic_energy = args['kinetic_energy']
        assert type(args['jacobian_norm2']) is float or args['jacobian_norm2'] is None
        self.jacobian_norm2 = args['jacobian_norm2']
        assert type(args['total_deriv']) is float or args['total_deriv'] is None
        self.total_deriv = args['total_deriv']
        assert type(args['directional_penalty']) is float or args['directional_penalty'] is None
        self.directional_penalty = args['directional_penalty']
        

        if self.layer_type == "blend" and not (self.time_length == 1.0 and self.train_T == False):
            raise ValueError("!! Setting time_length from None to 1.0 due to use of Blend layers.")
            
        self.dims = '-'.join([str(int(self.hdim_factor * data_dims))] * self.nhidden)
        self.args = args
    def __str__(self):
        return self.args




class Generator(nn.Module):
    def __init__(self, arg, data_dim):
        super(Generator, self).__init__()
        group1 = (arg.l1int is not None) + (arg.l2int is not None) + (arg.dl2int is not None) + (arg.JFrobint is not None) + (arg.JdiagFrobint is not None) + (arg.JoffdiagFrobint is not None)
        group2 = (arg.kinetic_energy is not None) + (arg.jacobian_norm2 is not None) + (arg.total_deriv is not None) + (arg.directional_penalty is not None)
        if (group1 > 0) and (group2 > 0):
            raise ValueError("regularizer group should be selected once")
        elif (group1 > 0):
            self.mode = 1
        else:
            self.mode = 2
        arg.mode = self.mode
        self.arg = arg
        regularization_fns, self.regularization_coeffs = create_regularization_fns(arg)
        self.model = build_model_tabular(arg, data_dim, regularization_fns)
    
    def compute_likelihood_loss(self, data):
        zero = torch.zeros(data.shape[0], 1).to(data)

        z, delta_logp = self.model(data, zero)  # run model forward

        logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)
        reg_loss = None
        if len(self.regularization_coeffs) > 0:
            reg_states = get_regularization(self.model, self.regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, self.regularization_coeffs) if coeff != 0
            )
            reg_loss = torch.abs(reg_loss).mean()
        return loss, reg_loss
        
    def forward(self, z):
        zero = torch.zeros(z.shape[0], 1).to(z)
        x = self.model(z, zero, reverse=True)  # run model Backward 
        return x[0]

    def restore_model(self, model, filename):
        checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpt["state_dict"])
        set_cnf_options(self.arg, self.model)
        return model

