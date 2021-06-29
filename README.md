# invertible GAN
## 1. Requirement
- python version : Python 3.7.7
- package information : requirements.txt

## 2. Train
1. train base model for '5.2. Experimental Results'
    ```
    example : 'sh train_base_main.sh veegan adult 1'

    first option : model [identity, privbn, clbn, independent, uniform, ctgan, medgan, tablegan, tvae, veegan]
    second option : data [adult, news]
    third option : GPU number to use <int>
    ```
2. train iGAN model for '5.2. Experimental Results'
    ```
    example : 'sh train_igan_main.sh adult L 1'

    first option : data [adult, news]
    second option : likelihood coef [L(Increase NLL), Q(Decrease NLL), 0(Only GAN)]
    third option : GPU number to use <int>
    ```
3. train iGAN model for '5.4. Sensitivity Analyses'
    - Sensitivity for $\gamma$
        ```
        example : 'sh train_igan_sens_coef.sh adult 0.05 1'

        first option : data [adult, news]
        second option : likelihood coef <float>
        third option : GPU number to use <int>
        ```
    - Sensitivity for dim($h$)
        ```
        example : 'sh train_igan_sens_emb.sh adult 128 1'

        first option : data [adult, news]
        second option : embedding size <int>
        third option : GPU number to use <int>
        ```

## 3. Test
1. Check Model Score : Table1 ~ 22 
    - Only the trained model scores are printed
    - Also, Check the result with tensorboard log 'last_result/runs/'
    
2. Draw Real-Fake Distance Distribution : Figure3
    - You can draw the plot after igan, igan(L), igan(Q) are trained.
    - The figure will be saved in 'dist_info/'

3. FBB Attack Score : Table23
    - FBB Attack Roc Auc scores of the trained model are printed.

```
example : 'sh train.sh score adult 0'

first option : test type  [score(check model score), distplot(draw real-fake distance distribution), fbb(fbb attack score)]
second option : data [adult, news]
third option : GPU number to use <int>
```
