# Machine_learning_codes_scratch
This provides a code for coding logistic regression from scratch.
This loss function gives rise to the ordinary least squares regression model. Our goal is to minimize this loss function;

\begin{equation}
\underset{\theta}{\min} \, \ell(\theta) = \dfrac{1}{2}\sum_{i=1}^{n}\big(h_{\theta} (x_{i})- y_{i}\big) ^{2} 
\end{equation}



For the purpose of this lesson, we will solve this using gradient descent and later on use a analytical method to solve this. We begin with an initial $\theta$ and we will repeatedly perform the updates;\\

\begin{equation}
\begin{split}
\theta \text{: Initial Condition} \\
\theta_{j} = \theta_{j-1} -\underbrace{\alpha \dfrac{\partial}{\partial \theta_{j-1}}\ell(\theta)}_{Gradient}\\
\end{split}
\end{equation}
where $\theta$ refers to the learning rate which also needs to be properly set.With this algorithm we repeatedly take step in the direction of steepest decrease of $\ell (\theta )$. In order to use this we need to calculate the partial derivative term on the right hand side. We will work this out in the case of a single training example and in that case we can neglect the sum in the definition of $\ell$

\begin{align}
	\begin{split}
	\dfrac{\partial}{\partial \theta}\ell(\theta) = 	\dfrac{\partial}{\partial \theta}\dfrac{1}{2}\big(h_{\theta}(x)-y\big) ^{2}\\
	=2.\dfrac{1}{2}\big(h_{\theta}(x)-y\big).\dfrac{\partial}{\partial\theta_{j}}\big(h_{\theta}(x)-y\big)\\
	=\big(h_{\theta(x)-y}\big).\dfrac{\partial}{\partial \theta_{j}}\bigg(\sum_{i=0}^{n} \theta_{i}x_{i}-y\bigg)\\
	=\big(h_\theta(x)-y\big)x_{j}
	\end{split}
\end{align}
For a single training example, we now have this as the update rule:

\begin{equation}
	\theta_{j}: \theta_{j-1}+\alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x_j^{(i)}
\end{equation}
