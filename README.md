# Invertible Table GAN
## 1. Requirement
- python version : Python 3.7.7
- package information : requirements.txt

## 2. Train
1. train ITGAN model
    ```
    example : 'python train_itgan.py --data --random_num --GPU_NUM --emb_dim --en_dim --d_dim --d_dropout --d_leaky --layer_type --hdim_factor --nhidden --likelihood_coef --gt --dt --lt --kinetic --kinetic_every_learn'
    
    data: dataset name
    random_num: random_seed to use
    GPU_NUM: GPU number to use
    emb_dim: $dim(h)$
    en_dim: $n_{e(r)}$ = 2 -> "256,128", 3 -> "512,256,128" 
    d_dim: $n_d$ = 2 -> "256,256", 3 -> "256,256,256" 
    d_dropout: a
    d_leaky: b
    layer_type: $blend[M_i(z,t) = t], simblenddiv1[M_i(z,t) = sigmoid(FC(z⊕t))]$
    hdim_factor: M
    nhidden: K
    likelihood_coef: $\gamma$
    gt: $period_G$
    dt: $period_D$
    lt: $period_L$
    kinetic: kinetic regularizer coef
    kinetic_every_learn: if 0 apply kinetic regularizer every G likelihood training, else every all G training
    
    the name of a file to be saved is combination of each parameter
    ```
    All parameter (except kinetic, kinetic_every_learn) is in Appendix D.
    kinetic: 0.1 for ITGAN(Q) of census and ITGAN(Q), ITGAN(L) for cabs, 1.0 for others  
    kinetic_every_learn: 1 for census, 0 for others

## 3. Test
1. Check Model Score : Table 1, 2, 3, 4, 5, 6, 8, 9
    - You can check the result in json file of last_result/score_info/{data} 
    - For Training Model(base5~9, ITGAN), the value of the key "best" is the model score
    - Also, Check the result with tensorboard log in 'last_result/runs/{data}'

2. FBB Attack Score : Table 7, 10
    - FBB Attack Roc Auc scores of the trained model are printed.
    ```
    example : 'python test_fbb.py --data --GPU_NUM --file --subopt'
    
    data: dataset name
    GPU_NUM: GPU number to use
    file: file name of model to use
    subopt: return subopt fbb result
    ```

## 4. License

Copyright (C) 2023 Samsung SDS Co., Ltd. All rights reserved.
Released under the Samsung SDS Public License V1.0.
For details on the scope of licenses, please refer to the License.md file (https://github.com/leejaehoon2016/ITGAN/blob/main/License.md).

