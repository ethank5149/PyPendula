# PyPendula
### Primary source code repository for PyPendula

let:

$$(x_1, y_1)=(      l\sin\theta_1,       l\cos\theta_1)$$
$$(x_2, y_2)=(x_1 + l\sin\theta_2, y_1 + l\cos\theta_2)$$
$$(x_3, y_3)=(x_2 + l\sin\theta_3, y_2 + l\cos\theta_3)$$

s.t.:

$$(\dot{x}_1, \dot{y}_1)=(            l\dot{\theta}_1\cos\theta_1,           -l\dot{\theta}_1\sin\theta_1)$$
$$(\dot{x}_2, \dot{y}_2)=(\dot{x}_1 + l\dot{\theta}_2\cos\theta_2, \dot{y}_1 -l\dot{\theta}_2\sin\theta_2)$$
$$(\dot{x}_3, \dot{y}_3)=(\dot{x}_2 + l\dot{\theta}_3\cos\theta_3, \dot{y}_2 -l\dot{\theta}_3\sin\theta_3)$$

and subsequently:

$$\begin{align*}
  ||v_1||^2 &= l^2\left(\dot{\theta}^2_1\cos^2\theta_1 + \dot{\theta}^2_1\sin^2\theta_1\right) \\
            &= l^2\dot{\theta}^2_1\left(\cos^2\theta_1 + \sin^2\theta_1\right) \\
            &= l^2\dot{\theta}^2_1 \\
\end{align*}$$

$$\begin{align*}
  ||v_2||^2 &= l^2\left[(\dot{\theta}_1\cos\theta_1 + \dot{\theta}_2\cos\theta_2)^2 + (\dot{\theta}_1\sin\theta_1 + \dot{\theta}_2\sin\theta_2)^2\right] \\
            &= l^2\left[(\dot{\theta}_1\cos\theta_1 + \dot{\theta}_2\cos\theta_2)^2 + (\dot{\theta}_1\sin\theta_1 + \dot{\theta}_2\sin\theta_2)^2\right] \\
\end{align*}$$

$$\begin{align*}
  ||v_3||^2 &= l^2\left[(\dot{\theta}_1\cos\theta_1 + \dot{\theta}_2\cos\theta_2 + \dot{\theta}_3\cos\theta_3)^2 + (\dot{\theta}_1\sin\theta_1 + \dot{\theta}_2\sin\theta_2 + \dot{\theta}_3\sin\theta_3)^2\right] \\
            &= l^2\left[(\dot{\theta}_1\cos\theta_1 + \dot{\theta}_2\cos\theta_2 + \dot{\theta}_3\cos\theta_3)^2 + (\dot{\theta}_1\sin\theta_1 + \dot{\theta}_2\sin\theta_2 + \dot{\theta}_3\sin\theta_3)^2\right] \\
\end{align*}$$
