import matplotlib.pyplot as plt
from abel.basex import basex_transform
from abel.tools.analytical import StepAnalytical

# This example calculates the BASEX transform of a step function and
# compares with the analtical result.
fig, ax = plt.subplots(1, 1)
plt.title('Abel tranforms of a step function')

n = 301
r_max = 50
A0 = 10.0
r1 = 6.0
r2 = 14.0

# define a symmetric step function and calculate its analytical Abel transform
st = StepAnalytical(n, r_max, r1, r2, A0)

ax.plot(st.r, st.func, 'b', label='Original signal')

ax.plot(
    st.r, st.abel*0.05, 'r',
    label='Direct Abel transform x0.05 [analytical]')

center = n//2
right_half = st.abel[center:]
left_half = st.abel[:center+1][::-1]
# BASEX Transform:
# Calculate the inverse abel transform for the centered data
recon_right = basex_transform(
    right_half, basis_dir='./', dr=st.dr, verbose=True)
recon_left = basex_transform(
    left_half, basis_dir='./', dr=st.dr, verbose=False)
plt.plot(
    st.r[center:], recon_right, '--.', c='red',
    label='Inverse transform [BASEX]')

plt.plot(st.r[:center+1], recon_left[::-1], '--.', c='red')

ax.legend()

ax.set_xlim(-20, 20)
ax.set_ylim(-5, 20)
ax.set_xlabel('x')
ax.set_ylabel("f(x)")

plt.legend()
plt.show()
