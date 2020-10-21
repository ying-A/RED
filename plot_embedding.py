import pickle
import torch
import numpy as np
import random
random.seed(43)
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
import matplotlib.pyplot as plt
f = open("transformer_analysis.pkl","rb")
# f = open("transformer_full_order_analysis.pkl","rb")
a = []
data = []
labels = []
try:
    while True:
        tmp = pickle.load(f)
        turn_length = len(tmp['his_turn_end_ids'][0])
        embs = torch.index_select(tmp['enc'][0],0,tmp['his_turn_end_ids'][0]).data.cpu().numpy()
        t = {}
        t['xs']=tmp['xs'][0].data.cpu().numpy()
        # t['enc']=tmp['enc'][0].data.cpu().numpy()
        # t['his_turn_end_ids']=tmp['his_turn_end_ids'][0].data.cpu().numpy()
        t['sen_emb'] = embs
        t['turn_length'] = turn_length
        t['ids'] = np.array([i for i in range(turn_length)])
        a.append(t)
        for emb in embs:
            data.append(emb)
        for label in t['ids']:
            labels.append(label)
except:
    pass
data_tmp = data
labels_tmp = labels
length = len(data)
ids = [i for i in range(length)]
random.shuffle(ids)
for i in ids:
    data_tmp[i] = data[i]
    labels_tmp[i] = labels[i]

data = data_tmp[:1000]
labels = labels_tmp[:1000]
print(len(data))
print(len(labels))
tsne = TSNE(n_components=2, init='pca', random_state=0)
data = tsne.fit_transform(data)

x_min, x_max = np.min(data, 0), np.max(data, 0)
data = (data - x_min) / (x_max - x_min)
cm_subsection = np.linspace(0,1, len(set(labels)))
colors = [ plt.cm.gist_rainbow(x) for x in cm_subsection ]


fig = plt.figure()
for i in range(data.shape[0]):
    plt.text(data[i, 0], data[i, 1], str(labels[i]),
                color=colors[labels[i]],
                fontdict={'size': 8})

plt.xticks([])
plt.yticks([])
cmap = mpl.cm.gist_rainbow_r
norm = mpl.colors.Normalize(vmin=0, vmax=len(set(labels)))
ax = fig.add_axes([0.125, 0.06, 0.775, 0.02])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb.set_ticks(np.linspace(0,20,21))  
cb.set_ticklabels(('20','19','18','17','16','15','14','13','12','11','10','9','8','7','6','5','4','3','2','1','0'))  
cb.ax.tick_params(labelsize=10)
plt.savefig("transformer_emb.pdf",bbox_inches='tight')