import matplotlib.pyplot as plt


def get_xs_ys(l_list):
    xs = []
    ys = []

    for index, line in enumerate(l_list[:206]):
        x, y = line.strip().split(' ')
        xs.append(int(x))
        ys.append(float(y))
    return xs, ys


# ========================= Train ====================
# gn_train_c_file = 'GN_2_16channel_train.log'
# gn_train_g_file = 'GN_2_16group_train.log'
# bn_train_file = 'BN_2_train.log'
#
#
# with open(gn_train_c_file) as fp:
#     line_list = fp.readlines()
# xs, ys = get_xs_ys(line_list)
# plt.plot(xs, ys, label='GN_2_16channel_train')
#
# with open(gn_train_g_file) as fp:
#     line_list = fp.readlines()
# xs, ys = get_xs_ys(line_list)
# plt.plot(xs, ys, label='GN_2_16group_train')
#
# with open(bn_train_file) as fp:
#     line_list = fp.readlines()
# xs, ys = get_xs_ys(line_list)
# plt.plot(xs, ys, label='BN_2_train')


# ============================== Test =======================
gn_train_c_file = 'GN_2_16channel_test.log'
gn_train_g_file = 'GN_2_16group_test.log'
bn_train_file = 'BN_2_test.log'


with open(gn_train_c_file) as fp:
    line_list = fp.readlines()
xs, ys = get_xs_ys(line_list)
plt.plot(xs, ys, label='GN_2_16channel_test')

with open(gn_train_g_file) as fp:
    line_list = fp.readlines()
xs, ys = get_xs_ys(line_list)
plt.plot(xs, ys, label='GN_2_16group_test')

with open(bn_train_file) as fp:
    line_list = fp.readlines()
xs, ys = get_xs_ys(line_list)
plt.plot(xs, ys, label='BN_2_test')

plt.legend()
# plt.show()
plt.savefig('2_test_loss.png')