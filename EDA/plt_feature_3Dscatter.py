import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# plt-3D
train['3D_axis1'] = train.loc[:,['open_count_last_10_days']]
train['3D_axis2'] = train.loc[:,['open_count_last_30_days']]
train['3D_axis3'] = train.loc[:,['open_count_last_60_days']]

ax = plt.figure(figsize = (16,10)).gca(projection='3d')
ax.view_init(30, -20)
ax.scatter(
    xs = train.loc[:,'3D_axis1'],
    ys = train.loc[:,'3D_axis2'],
    zs = train.loc[:,'3D_axis3'],
    c = train.loc[:,'open_flag'],
    cmap = 'tab10')
ax.set_xlabel('3D_axis1')
ax.set_ylabel('3D_axis2')
ax.set_zlabel('3D_axis3')
plt.show


ax = plt.figure(figsize = (16,10)).gca(projection='3d')
ax.view_init(30, -110)
ax.scatter(
    xs = train.loc[:,'3D_axis1'],
    ys = train.loc[:,'3D_axis2'],
    zs = train.loc[:,'3D_axis3'],
    c = train.loc[:,'open_flag'],
    cmap = 'tab10')
ax.set_xlabel('3D_axis1')
ax.set_ylabel('3D_axis2')
ax.set_zlabel('3D_axis3')
plt.show
