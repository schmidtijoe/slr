import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_relaxation_analytic(ux, uy, uz, mx, my, mz, me, phi, t1_sec, t2_sec, dt_sec):
    """
    Calculates the effect of rotation and relaxation matrices without the use of dot and matrix array creation
    Parameters
    ----------
    ux : x component of vector about which to rotate
    uy : y component of vector about which to rotate
    uz : z component of vector about which to rotate
    mx : x component of magnetization vector to rotate
    my : y component of magnetization vector to rotate
    mz : z component of magnetization vector to rotate
    me : e component of magnetization vector to rotate
    phi : angle of rotation
    t1_sec : t1 relaxation
    t2_sec : t2 relaxation
    dt_sec : time step

    Returns magnetization vector
    -------

    """
    e1 = np.exp(-dt_sec / t1_sec)  # T1 never 0 or below in simulation parameters
    e2 = np.exp(-dt_sec / t2_sec)  # T2 never 0 or below (should incorporate case anyhow?!)
    co = np.cos(phi)
    si = np.sin(phi)
    a = 1.0 - co
    return np.transpose([
        (mx * (co + a * ux ** 2) + my * (a * ux * uy - uz * si) + mz * (a * ux * uz + uy * si)) * e2,
        (mx * (a * ux * uy + uz * si) + my * (co + a * uy ** 2) + mz * (a * uy * uz - ux * si)) * e2,
        (mx * (a * ux * uz - uy * si) + my * (a * uy * uz + ux * si) + mz * (co + a * uz ** 2)) * e1 + (1 - e1) * me,
        me
    ])


def spinor_rotation(phi, nx, ny, nz, m_p, m_m, mz, me):
    alpha = np.cos(phi / 2) - 1j * nz * np.sin(phi / 2)
    beta = -1j * (nx + 1j * ny) * np.sin(phi/2)
    return np.transpose([
        np.conj(alpha)**2 * m_p - beta**2 * m_m + 2 * np.conj(alpha) * beta * mz,
        - np.conj(beta)**2 * m_p + alpha**2 * m_m + 2* alpha * np.conj(beta) * mz,
        - np.conj(alpha * beta) * m_p - (alpha * beta) * m_m + (alpha * np.conj(alpha) - beta * np.conj(beta)) * mz,
        me
        ])


# Create magnetization vector
m_init = np.array([0, 0, 1, 1])

m_exc_mag = rotation_relaxation_analytic(0, 1, 0, *m_init, np.pi / 50, 1.5, 1, 5*1e-6)

m_exc_spinor = spinor_rotation(np.pi/50, 0, 1, 0, *m_init)
m_exc_spinor = np.array([np.real(m_exc_spinor[0] + m_exc_spinor[1]) / 2, np.imag(m_exc_spinor[0] - m_exc_spinor[1]) / 2,
                         m_exc_spinor[2], m_exc_spinor[3]], dtype=float)

origin = np.array([0, 0, 0])

fig = plt.figure(figsize=(10, 7), dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-0.1, 1.1)
ax.quiver(*origin, *m_exc_mag[:-1], color='#32a852', label='magnetization domain computation')
ax.quiver(*origin, *m_exc_spinor[:-1], color='#b02d10', label='spin domain computation')
plt.legend()
plt.show()





