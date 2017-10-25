import sys
from numpy.random import standard_normal
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap

# y = standard_normal((5, 1))
# x = standard_normal((5,1))
# z = standard_normal(5)
# bs = StationaryBootstrap(3, x, y=y, z=z)
# for data in bs.bootstrap(100):
#     bs_x = data[0][0]
#     bs_y = data[1]['y']
#     bs_z = bs.z


y = standard_normal((5, 1))
x = standard_normal((5, 2))
z = standard_normal(5)
bs = CircularBlockBootstrap(3, x, y=y, z=z)
for data in bs.bootstrap(100):
    print data; sys.exit()
    bs_x = data[0][0]
    bs_y = data[1]['y']
    bs_z = bs.z