import warnings, random, copy, os, sys, datetime, argparse
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, Sequential, TripletMarginLoss, Sigmoid
from torch.nn import functional as F

from synthesizers.AEModel import Encoder, Decoder, loss_function
from synthesizers.GeneratorModel import Generator, argument
from synthesizers.DiscriminatorModel import Discriminator

from util.base import BaseSynthesizer
from util.transformer_new import BGMTransformer
from util.evaluate import compute_scores, _compute_for_distribution
from util.evaluate_cluster import compute_cluster_scores
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save

from util.data import load_dataset
from tensorboardX import SummaryWriter
from util.benchmark import benchmark

artificial_data = ["alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
short_real_data = ["adult", "news", "cabs", "king", "airbnb", "bank2", "safe"]
other_real_data = ["census", "covertype", "intrusion"]


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_grad, pac=1):
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_grad
    return gradient_penalty

class AEGANSynthesizer(BaseSynthesizer):
    def __init__(self, rtol, atol, batch_size, epochs, random_num, GPU_NUM, save_loc, save_arg, data_name, test_name,
                G_model, embedding_dim, G_args, G_lr, G_beta, G_l2scale, G_l1scale, 
                likelihood_coef, likelihood_learn_start_score, likelihood_learn_term,
                D_model, dis_dim, lambda_grad, D_lr, D_beta, D_l2scale,
                En_model, compress_dims, AE_lr, AE_beta, AE_l2scale,
                De_model, decompress_dims, L_func, loss_factor, sigma_coef, train=True):

        self.G_model=G_model; self.embedding_dim=embedding_dim; self.G_args=G_args; self.G_lr=G_lr; self.G_beta=G_beta; self.G_l2scale=G_l2scale
        self.G_l1scale = G_l1scale

        self.likelihood_coef = likelihood_coef; self.likelihood_learn_start_score = likelihood_learn_start_score
        self.likelihood_learn_term = likelihood_learn_term

        self.D_model=D_model; self.dis_dim=dis_dim; self.lambda_grad=lambda_grad; self.D_lr=D_lr; self.D_beta=D_beta; self.D_l2scale=D_l2scale

        self.En_model=En_model; self.compress_dims=compress_dims; self.AE_lr=AE_lr; self.AE_beta=AE_beta; self.AE_l2scale=AE_l2scale
        self.De_model=De_model; self.decompress_dims=decompress_dims; self.L_func=L_func; self.loss_factor = loss_factor ; self.sigma_coef = sigma_coef

        self.rtol=rtol; self.atol=atol; self.batch_size=batch_size; self.epochs=epochs; self.random_num=random_num ; self.save_loc = save_loc
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPU_NUM = GPU_NUM ; self.save_arg = save_arg ; self.data_name = data_name ; self.test_name = test_name

        if train:
            self.save_arg["excute_time"] = str(datetime.datetime.now())
            with open(self.save_loc + "/param/" + self.data_name + "/" + self.test_name + ".txt","a") as f:
                f.write("excute_time: " + self.save_arg["excute_time"] + "\n")
            self.writer = SummaryWriter(self.save_loc + "/runs/" + self.data_name + "/" + self.test_name)

        # torch.cuda.set_device(self.device)

        # for random
        fix_random_seed(self.random_num)
        
    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.train = train_data.copy()
        self.transformer = BGMTransformer(meta_data, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        
        self.test = test_data
        self.meta = meta_data
        
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_dim = self.transformer.output_dim

        encoder = self.En_model(data_dim, self.compress_dims, self.embedding_dim, self.sigma_coef).to(self.device)
        self.decoder = self.De_model(self.embedding_dim, self.decompress_dims, data_dim).to(self.device)
        self.generator = self.G_model(self.G_args, self.embedding_dim).to(self.device)
        self.discriminator = self.D_model(self.embedding_dim, self.dis_dim).to(self.device)

        optimizerAE = optim.Adam(list(encoder.parameters()) + list(self.decoder.parameters()),
                                 lr = self.AE_lr, betas= self.AE_beta, weight_decay= self.AE_l2scale)
        optimizerG = optim.Adam(self.generator.parameters(), lr = self.G_lr, betas= self.G_beta, weight_decay = self.G_l2scale)
        optimizerD = optim.Adam(self.discriminator.parameters(), lr = self.D_lr, betas= self.D_beta, weight_decay = self.D_l2scale)
                    

        iter = 0
        best_model_dict = model_save_dict(self.meta["problem_type"])
        every_model_dict = {"name": "AEGANSynthesizer", "arg" : self.save_arg, "model": []}
        
        mean_z = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std_z = mean_z + 1

        for i in range(self.epochs):
            for id_, data in enumerate(loader): 
    
                iter += 1

                ######## Real, fake Encoding data Generation #########
                real = data[0].to(self.device)
                mu, _, _ = encoder(real)
                emb = mu

                fakez = torch.normal(mean=mean_z, std=std_z)
                fake_h = self.generator(fakez)

                real_h = emb

                ######## update discriminator #########
                y_fake = self.discriminator(fake_h)
                y_real = self.discriminator(real_h)

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(self.discriminator, real_h, fake_h, self.device, self.lambda_grad)

                
                loss_d = loss_d + pen 
                optimizerD.zero_grad()

                loss_d.backward()
                optimizerD.step()

                ######### update generator with W-GAN Loss ##########
                fakez = torch.normal(mean=mean_z, std=std_z)
                fake_h = self.generator(fakez)

                y_fake = self.discriminator(fake_h)

                loss_g = -torch.mean(y_fake)

                reg_g = 0
                if self.G_l1scale is not None and self.G_l1scale != 0:
                    reg_g = self.G_l1scale * sum([i.abs().sum() for i in self.generator.parameters()])
                optimizerG.zero_grad()
                (loss_g + reg_g).backward()
                optimizerG.step()
                

                ######## update generator with Likelihood Loss ##########

                real = data[0].to(self.device)
                mu, _, _ = encoder(real)
                emb = mu
                real_h = emb
                likelihood_loss, likelihood_reg_loss = self.generator.compute_likelihood_loss(real_h)
                self.writer.add_scalar('losses/likelihood_loss', likelihood_loss, iter)
                if (likelihood_reg_loss is not None):
                    self.writer.add_scalar('losses/likelihood_reg_loss', likelihood_reg_loss, iter)

                likelihood_cal = True
                if (likelihood_reg_loss is not None and self.likelihood_coef != 0):
                    last_like_loss = likelihood_loss * self.likelihood_coef + likelihood_reg_loss
                elif (likelihood_reg_loss is not None):
                    last_like_loss = likelihood_reg_loss
                elif (self.likelihood_coef != 0):
                    last_like_loss = likelihood_loss * self.likelihood_coef
                else:
                    likelihood_cal = False
                


                if (self.likelihood_learn_start_score is None or best_model_dict["f1"]["f1"] > self.likelihood_learn_start_score) and \
                    iter % self.likelihood_learn_term == (self.likelihood_learn_term - 1):
                    if (likelihood_cal == True):
                        reg_g = 0
                        if self.G_l1scale is not None and self.G_l1scale != 0:
                            reg_g += self.G_l1scale * sum([i.abs().sum() for i in self.generator.parameters()])
                        optimizerG.zero_grad()
                        (last_like_loss + reg_g).backward()
                        optimizerG.step()
                ###########################
                elif (likelihood_reg_loss is not None):
                    reg_g = 0
                    if self.G_l1scale is not None and self.G_l1scale != 0:
                        reg_g += self.G_l1scale * sum([i.abs().sum() for i in self.generator.parameters()])
                    optimizerG.zero_grad()
                    (likelihood_reg_loss + reg_g).backward()
                    optimizerG.step()
                ##########################
                    

                ########## update vae ###########
                real = data[0].to(self.device)
                mu, _, _ = encoder(real)
                emb = mu

                fakez = torch.normal(mean=mean_z, std=std_z)
                fake_h = self.generator(fakez)
                real_h = emb

                y_fake = self.discriminator(fake_h)
                y_real = self.discriminator(real_h)
                
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = self.L_func(
                    rec, real, sigmas, mu, 0, self.transformer.output_info, self.loss_factor)
                
                rec_syn, sigmas_syn = self.decoder(fake_h)
                mu_syn, _, _ = encoder(rec_syn)
                emb_syn = mu_syn

                loss_gan = -torch.mean(y_real) + ((emb_syn - fake_h) ** 2).mean()
                loss_ae = loss_1 + loss_2 + loss_gan
                
                optimizerAE.zero_grad()
                loss_ae.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                ######### plotting #########
                self.writer.add_scalar('losses/G_loss', loss_g, iter)
                self.writer.add_scalar('losses/D_loss', loss_d, iter)
                self.writer.add_scalar('losses/AE_loss', loss_ae, iter)


            if True: # compute scores every epochs (# if i >= 150 and i % 2 == 0:)
                syn_data = self.sample(self.train.shape[0])
                score = compute_scores(self.train, self.test, syn_data, self.meta) 
                s = score.loc[0].index.to_list()
                for k in range(1, score.shape[1]):
                    self.writer.add_scalar('average/'+s[k], score.iloc[:, k].mean(), i)
                    for j in range(len(score)):
                        self.writer.add_scalar(score['name'][j]+'/'+s[k], score.iloc[j, k], i)
                
                if "likelihood" not in self.meta["problem_type"]:
                    cluster_score = compute_cluster_scores(self.train, self.test, syn_data, self.meta) 
                    s = cluster_score.loc[0].index.to_list()
                    for k in range(1, cluster_score.shape[1]):
                        self.writer.add_scalar('average/'+s[k], cluster_score.iloc[:, k].mean(), i)
                        for j in range(len(cluster_score)):
                            self.writer.add_scalar(cluster_score['name'][j]+'/'+s[k], cluster_score.iloc[j, k], i)
                    change, best_model_dict = model_score_save(self.meta["problem_type"], pd.concat([score.mean(), cluster_score.mean()],axis=0), 
                                                                best_model_dict, i,True)
                else:
                    change, best_model_dict = model_score_save(self.meta["problem_type"], score.mean(), best_model_dict, i,True)

                
                if change == True:
                    torch.save(best_model_dict, self.save_loc + "/score_info/"+ self.data_name + "/" + self.test_name + '.json')

                every_model_dict["model"].append({'generator': copy.deepcopy(self.generator.state_dict()), "decoder" : copy.deepcopy(self.decoder.state_dict())})    
                torch.save(every_model_dict, self.save_loc + "/save_model/"+ self.data_name + "/" + self.test_name + '.pth')
                

    def sample(self, n, z_vector = False):
        self.generator.eval()
        self.decoder.eval()

        mean_z = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std_z = mean_z + 1

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            fakezs = torch.normal(mean=mean_z, std=std_z)
            fake = self.generator(fakezs)
            fake, _  = self.decoder(fake)
            fake = apply_activate(fake, self.transformer.output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        self.generator.train()
        self.decoder.train()
        result = self.transformer.inverse_transform(data, None)
        if z_vector:
            return result, fakezs
        else:
            return result

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])

    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        
        self.train = train_data.copy()
        self.transformer = BGMTransformer(meta_data, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        
        self.test = test_data
        self.meta = meta_data
        
        data_dim = self.transformer.output_dim
        
        self.decoder = self.De_model(self.embedding_dim, self.decompress_dims, data_dim).to(self.device)
        self.generator = self.G_model(self.G_args, self.embedding_dim).to(self.device)
        if "name" in checkpoint:
            self.generator.load_state_dict(checkpoint["model"][choosed_model]['generator'])
            self.decoder.load_state_dict(checkpoint["model"][choosed_model]['decoder'])
            self.generator.eval()
            self.decoder.eval()
        else:
            self.generator.load_state_dict(checkpoint["info"][choosed_model]['generator'])
            self.decoder.load_state_dict(checkpoint["info"][choosed_model]['decoder'])
            self.generator.eval()
            self.decoder.eval()



# Commented when testing
if __name__ == "__main__":
    ################################################ Default Value #######################################################
    ## basic info
    data = "adult" ; test_name = "iGAN"
    rtol = 1e-3 ; atol = 1e-3; batch_size = 2000 ; epochs = 500 ; random_num = 777 ; GPU_NUM = 0 ; save_loc= "last_result"
    G_model= Generator; embedding_dim= 128; G_lr= 2e-4; G_beta= (0.5, 0.9); G_l2scale= 1e-6 ; G_l1scale = 1e-6
    likelihood_coef = 0 ; likelihood_learn_start_score = None ; likelihood_learn_term = 6
    D_model= Discriminator; dis_dim= (256, 256); lambda_grad= 10; D_lr= 2e-4; D_beta= (0.5, 0.9); D_l2scale= 1e-6
    En_model= Encoder; compress_dims= (256, 128); AE_lr= 2e-4; AE_beta= (0.5, 0.9); AE_l2scale= 1e-6
    De_model= Decoder; decompress_dims= (128, 256); L_func= loss_function; loss_factor = 2 ; sigma_coef = 0

    # Generator CNF info
    layer_type = "blend" # layer type ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    hdim_factor = 1. # hidden layer size <int>
    nhidden = 3 # the number of hidden layers <int>
    num_blocks = 1 # the number of ode block(cnf statck) <int>
    time_length = 1.0 # time length(if blend type is choosed time length has to be 1.0) <float>
    train_T = False # Traing T(if blend type is choosed this has to be False)  [True, False]
    divergence_fn = "approximate" # how to calculate jacobian matrix ["brute_force", "approximate"]
    nonlinearity = "tanh" # the act func to use # ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
    test_solver = solver = "dopri5" # ode solver ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
    test_atol = atol  # <float>
    test_rtol = rtol  # <float>
    step_size = None # "Optional fixed step size." <float, None>
    first_step = 0.166667 # only for adaptive solvers  <float> 사용x
    residual = True  # use residual net odefunction [True, False]
    rademacher = False # rademacher or gaussian [True, False]
    batch_norm = False  # use batch norm [True, False]
    bn_lag = 0. # batch_norm of bn_lag <float>
    adjoint = True

    # regularizer of odefunction(you must use either regularizer group 1 or regularizer group 2, not both)
    # regularizer group 1
    l1int = None # "int_t ||f||_1" <float, None>
    l2int = None # "int_t ||f||_2" <float, None>
    dl2int = None # "int_t ||f^T df/dt||_2"  <float, None>
    JFrobint = None # "int_t ||df/dx||_F"  <float, None>
    JdiagFrobint = None # int_t ||df_i/dx_i||_F  <float, None>
    JoffdiagFrobint = None # int_t ||df/dx - df_i/dx_i||_F  <float, None>

    # regularizer group 2
    kinetic_energy = 1. # int_t ||f||_2^2 <float, None>
    jacobian_norm2 = 1. # int_t ||df/dx||_F^2 <float, None>
    total_deriv = None # int_t ||df/dt||^2 <float, None>
    directional_penalty = None  # int_t ||(df/dx)^T f||^2 <float, None>
    ################################################ Default Value #######################################################
    
    parser = argparse.ArgumentParser('iGAN')
    
    parser.add_argument('--layer_type', type=str)
    parser.add_argument('--rademacher', type=str)
    parser.add_argument('--likelihood_coef', type=float)
    parser.add_argument('--adjoint', type=str)
    parser.add_argument('--residual', type=str)

    
    # parser.add_argument('--nonlinearity', type=str, default = "tanh")
    # parser.add_argument('--kinetic', type=float, default = 1.)
    # parser.add_argument('--num_blocks', type=int, default = 1)
    # parser.add_argument('--hdim_factor', type=float, default = 1.)
    # parser.add_argument('--nhidden', type=int, default = 3)
    # parser.add_argument('--data', type=str, default = 'covtype')
    # parser.add_argument('--test_name', type=str, default = 'iGAN')
    # parser.add_argument('--G_l1scale', type=float, default = 1e-6)
    # parser.add_argument('--G_l2scale', type=float, default = 1e-6)
    # parser.add_argument('--likelihood_learn_term', type=int, default = 6)
    # parser.add_argument('--random_num', type=int, default = 777)
    # parser.add_argument('--embedding_dim', type=int, default = 128)
    # parser.add_argument('--GPU_NUM', type = int, default = 0)
    

    arg_of_parser = parser.parse_args()

    # test_case
    data = 'covtype'
    layer_type = arg_of_parser.layer_type
    rademacher = False if arg_of_parser.rademacher == "False" else True
    likelihood_coef = arg_of_parser.likelihood_coef
    adjoint = False if arg_of_parser.adjoint == "False" else True
    residual = False if arg_of_parser.residual == "False" else True
    test_name = "_".join([str(i) for i in [layer_type, rademacher, likelihood_coef, adjoint, residual]])

    # layer_type = arg_of_parser.layer_type
    # hdim_factor = arg_of_parser.hdim_factor
    # nhidden = arg_of_parser.nhidden
    # kinetic_energy = jacobian_norm2 = arg_of_parser.kinetic
    # num_blocks = arg_of_parser.num_blocks
    # nonlinearity = arg_of_parser.nonlinearity
    # rademacher = False if arg_of_parser.rademacher == "False" else True
    # likelihood_coef = arg_of_parser.likelihood_coef
    # test_name = "_".join([str(i) for i in [layer_type, hdim_factor, nhidden, kinetic_energy, num_blocks, nonlinearity, rademacher, likelihood_coef]])

    # G_l2scale = arg_of_parser.G_l2scale
    # G_l1scale = arg_of_parser.G_l1scale
    # likelihood_learn_term = arg_of_parser.likelihood_learn_term
    # random_num = arg_of_parser.random_num
    # embedding_dim = arg_of_parser.embedding_dim
    # GPU_NUM = arg_of_parser.GPU_NUM
    


    G_args = {
        'layer_type' : layer_type,
        'hdim_factor' : hdim_factor,
        'nhidden' : nhidden,
        'num_blocks' : num_blocks,
        'time_length' : time_length,
        'train_T' : train_T,
        'divergence_fn' : divergence_fn,
        'nonlinearity' : nonlinearity,
        'solver' : solver,
        'atol' : atol,
        'rtol' : rtol,
        'test_solver' : test_solver,
        'test_atol' : test_atol,
        'test_rtol' : test_rtol,
        'step_size' : step_size,
        'first_step' : first_step,
        'residual' : residual,
        'rademacher' : rademacher,
        'batch_norm' : batch_norm,
        'bn_lag' : bn_lag,
        'l1int' : l1int,
        'l2int' : l2int,
        'dl2int' : dl2int,
        'JFrobint' : JFrobint,
        'JdiagFrobint' : JdiagFrobint,
        'JoffdiagFrobint' : JoffdiagFrobint,
        'kinetic_energy' : kinetic_energy,
        'jacobian_norm2' : jacobian_norm2,
        'total_deriv' : total_deriv,
        'directional_penalty' : directional_penalty,
        'adjoint' : adjoint}

    arg = {"rtol":rtol,
            "atol":atol,
            "batch_size":batch_size,
            "epochs":epochs,
            "random_num":random_num,
            "GPU_NUM":GPU_NUM,
            "save_loc":save_loc,
            "test_name":test_name,
            "data_name": data,
            "G_model":G_model,
            "embedding_dim":embedding_dim,
            "G_args" : argument(G_args, embedding_dim),
            "G_lr":G_lr,
            "G_beta":G_beta,
            "G_l2scale":G_l2scale,
            "G_l1scale":G_l1scale,
            "likelihood_coef":likelihood_coef,
            "likelihood_learn_start_score":likelihood_learn_start_score,
            "likelihood_learn_term":likelihood_learn_term,
            "D_model":D_model,
            "dis_dim":dis_dim,
            "lambda_grad":lambda_grad,
            "D_lr":D_lr,
            "D_beta":D_beta,
            "D_l2scale":D_l2scale,
            "En_model":En_model,
            "compress_dims":compress_dims,
            "AE_lr":AE_lr,
            "AE_beta":AE_beta,
            "AE_l2scale":AE_l2scale,
            "De_model":De_model,
            "decompress_dims":decompress_dims,
            "L_func":L_func,
            "loss_factor":loss_factor,
            "sigma_coef" : sigma_coef}
    
    arg["save_arg"] = arg.copy()
    mkdir(save_loc, data)
    with open(save_loc + "/param/"+ data + "/" + test_name + '.txt',"a") as f:
        f.write(data + " AEGANSynthesizer" + "\n")
        f.write(str(arg) + "\n")
        f.write(str(G_args) + "\n")
    
    a,b = benchmark(AEGANSynthesizer, arg, data)

