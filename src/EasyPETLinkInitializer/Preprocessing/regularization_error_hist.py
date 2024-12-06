import numpy as np
import matplotlib.pyplot as plt



top = 0.9
distance = np.linspace(0, 60,30)
X, Y = np.meshgrid(distance,distance)

desvio = distance*np.tan(np.deg2rad(top))



# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(X, Y,desvio,50)
# plt.show()
print(desvio)



def f(x, y,top):
    return np.sqrt(x**2+y**2)*np.tan(np.deg2rad(top))
    # return x*np.tan(np.deg2rad(top))

top = 0.9
x = np.abs(np.arange(-30, 36,6))+30
y = np.abs(np.arange(-30, 36,6))

X, Y = np.meshgrid(x, y)
Z = f(X, Y,top)

fig = plt.figure()
ax = plt.axes()
ax.imshow(Z, cmap="jet")

Z = np.round(Z,2)
for (j,i),label in np.ndenumerate(Z):
    ax.text(i,j,label,ha='center',va='center')
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50)
# ax.contour3D(X, Y, Z, 50)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');

micro = np.arange(0, 8)
angles = 0.9/2**micro
print(angles)
distance = np.linspace(0, 60, 10)
inv_distance = np.abs(np.linspace(-60, 0, 10))
i = 0
error = [None]*len(angles)
error_mean = [None]*len(angles)
for angle in angles:
    error[i] = (distance*np.tan(np.deg2rad(angle)))
    error_mean[i] = ((inv_distance*np.tan(np.deg2rad(angle)) + error[i])/2)[0]
    i += 1
    # print(error.max())
    print(error_mean)

fsize = 15
tsize = 18

tdir = 'in'

major = 5.0
minor = 3.0

# style = 'default'
# import matplotlib
# matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')
#
# plt.style.use(style)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = fsize
# plt.rcParams['legend.fontsize'] = tsize
# plt.rcParams['xtick.direction'] = tdir
# plt.rcParams['ytick.direction'] = tdir
# plt.rcParams['xtick.major.size'] = major
# plt.rcParams['xtick.minor.size'] = minor
# plt.rcParams['ytick.major.size'] = major
# plt.rcParams['ytick.minor.size'] = minor

error = np.array(error)


plt.figure()
plt.style.use("seaborn-dark")
plt.plot(angles, error_mean, '.-',  color='#4361ee', label="Mean error")
plt.plot(angles, error, '-',  color='#4361ee', alpha=0.2)
for err,d in zip(error[0],distance):
    print(err)
    plt.text(0.85, err.max(), "{} mm".format(int(np.round(d,0))), style='normal',fontsize=10)

# plt.fill_between(angles, np.zeros(len(angles)), np.max(error, axis=1),alpha=0.3, facecolor='#08F7FE')
plt.xlabel("Top Motor angle variation (º)")
plt.ylabel("Positional error (mm)")
plt.tick_params(top=True, right=True, which='both', direction='in')
plt.grid()
plt.legend()

plt.show()

