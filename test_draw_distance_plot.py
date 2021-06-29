import torch, argparse, os
import pandas as pd
import numpy as np
import seaborn as sns
from util.model_load import model_load
import matplotlib.pyplot as plt
from util.data import load_dataset
from util.evaluate import _compute_for_distribution

def ch(name):
    if "ctgan" in name:
        name = "ctgan"
    elif "tvae" in name:
        name = "tvae"
    elif "-" in name:
        name = "iGAN(L)"
    elif len(name.split("_")) < 2:
        pass
    elif float(name.split("_")[-2]) == 0.0:
        name = "iGAN"
    else:
        name = "iGAN(Q)"
    return name.replace(".npy","")
def draw_plot(result,name):
    color_dictionary = { "iGAN(L)" : "red", "iGAN(Q)" : "blue", "iGAN" : "black"}
    ax = sns.kdeplot(data=result, x = "Distance", hue="Model", fill=True, palette = color_dictionary, legend=True,
                clip = (0,float("inf")))#, kind="kde" , fill=True, legend = True
    plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='0')
    plt.xlim(0,result["Distance"].max())
    plt.xlabel('Real-Fake Distance', fontsize = 15)
    plt.ylabel("Probabilisy Density Function",fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig('plot_dist/{}.pdf'.format(name), dpi=300, bbox_inches='tight')
    plt.close()
dic = {
    "king" : [
        "1_5_1.5_3_256,256_512,256,128_0.2_0.5_1.0_False_0.0_888",
        "1_5_2.0_5_256,256_256,128_0.2_0.5_1.0_False_0.014_999",
        "blend_1_1_1.5_5_256,256,256_512,256,128_0.2_0.5_1.0_False_-0.011_999",
    ],

    "credit2" : [
        "3_1_1.5_3_256,256_256,128_0.2_0.0_1.0_False_-0.014_444",
        "simblenddiv1_5_5_1.5_3_256,256_256,128_0.2_0.0_1.0_False_0.05_666",
        "3_3_1.5_3_256,256_256,128_0.2_0.0_1.0_False_0.0_888",
    ],

    "cabs" : [
        "3_3_1.5_3_256,256_256,128_0.2_0.0_1.0_False_0.014_666",
        "3_1_1.5_3_256,256_256,128_0.0_0.0_0.1_False_-0.014_666",	
        "3_3_1.5_3_256,256_256,128_0.0_0.0_1.0_False_0.0_888"
    ],
}
if __name__ == "__main__":
    data_name = "credit2"
    for data_name in ["king","credit2"]:
        print(data_name)
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(data_name, benchmark=True)
        mc = dic[data_name]
        dist_lst = []
        for i in os.listdir(f"data_for_fbb/{data_name}"):
            print(mc)
            flag = True
            for m in mc:
                if m in i:
                    flag = False
            if flag:
                continue
            print(i)
            name = ch(i)
            syn_data = np.load(f"data_for_fbb/{data_name}/{i}")
            dist = _compute_for_distribution(train_data, test_data, syn_data, meta_data)
            tmp_dist = pd.DataFrame(dist, columns = ["Distance"])
            tmp_dist["Model"] = name
            dist_lst.append(tmp_dist)
        result = pd.concat(dist_lst)
        draw_plot(result,data_name)

