import seaborn as sns
import matplotlib.pyplot as plt
def plot_feature_scatter(df, cla_col):
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    
    sns.set_style("whitegrid")
    plt.figure()
    fig, ax = plt.subplots(4, 4, figsize=(20, 28))

    cla_val = df[cla_col].unique()
    features = [col for col in df.columns if col != cla_col]
    plt_n = len(features)
    for y ,cla_ in enumerate(cla_val):
      df_plt = df.loc[df[cla_col] == cla_]
      plt_ind_ = 1
      for j,feature_j in enumerate(features):
        for i,feature_i in enumerate(features):
            plt.subplot(plt_n, plt_n, plt_ind_)
            plt.scatter(df_plt.loc[:,feature_i], df_plt.loc[:,feature_j], marker=".", color=colors[y], alpha=0.2)
            plt.xlabel(feature_i, fontsize=9)
            plt.ylabel(feature_j, fontsize=9)
            plt_ind_ += 1
    plt.show()  
    

if __name__ == "__main__":
  cols_ = [col_ for col_ in df.columns]
  plot_feature_scatter(df.loc[:,cols_], cla_col="Y")
