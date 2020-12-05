## Supervised Learning Approach to Inverse Kinematics\n",
    "\n",
    "### Michael Menaker, Leo Ling\n",
    "\n",
    "Over the course of this project, we demonstrated the use of neural networks to solve the inverse kinematics problem. The inverse kinematics problem refers to the finding the movements required to reach a certain Cartesian configuration. Or, more formally, for a given joint space $Q$ and a Cartesian space $W$\n",
    "\n",
    "$F(Q)=W$ is the 1 to 1 mapping from joint positions/angles to a Cartesian coordinate, aka forward kinematics.\n",
    "\n",
    "Finding the inverse mapping from $W$ to $Q$, such that $F^{-1}(W)=Q$, is the inverse kinematics problem.\n",
    "\n",
    "For this project, we have elected to train a neural networks to approximate the inverse kinematic function for a relatively simple 3-Link System on a 2D plane. For some simple cases, there are analytical solutions or techniques to find the inverse function. Our work here is an example that this class of problem can be approximate using neural networks. This work closely follows a paper by [Duka](https://core.ac.uk/download/pdf/82122272.pdf), with the following parameters.\n",
    "\n",
    "1. $l_1=l_2=l_3=2$, The links all have the same size\n",
    "2. $q_1 \\in [0,\\pi]$, $q_2 \\in [-\\pi,0]$ , $q_3 \\in [-\\pi/2,\\pi/2]$, Each rotational joint movement is limited\n",
    "\n",
    "![3-Link Planar Manipulator](manipulator.png)\n"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "## Note that some code has been cut from the report and some images rescaled for brevity. \n",
    "To generate the training dataset for supervised learning, we constructed a meshgrid over the domain of the joint rotations, $q1, q2, q3$, and found the output Cartesian coordinate, labeled $E$ on the above figure. For supervised training of the neural network, the Cartesian coordinates and the end angle of the system, $\\Theta_E$, are the inputs. With the output being the three joint angles that produced this movement."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "executionInfo": {
     "elapsed": 537,
     "status": "ok",
     "timestamp": 1606506994865,
     "user": {
      "displayName": "Leo Ling",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhkCy_59QRiX50I6a8rcexp3HHvsDFtpk9mAxws=s64",
      "userId": "05198425524455268493"
     },
     "user_tz": 360
    },
    "id": "K3M-pCLRRzN8"
   },
   "outputs": [],
   "source": [
    "## Generate Data for an arbitrary three rotational joint system in 2D\n",
    "def x_e(q, l=l) :\n",
    "    sumX = 0\n",
    "    for ii in range(l.size):\n",
    "        sumX += l[ii] * np.cos(np.sum(q[:ii, :], axis=0))\n",
    "    return sumX\n",
    "def y_e(q, l=l) :\n",
    "    sumY = 0\n",
    "    for ii in range(l.size):\n",
    "        sumY += l[ii] * np.sin(np.sum(q[:ii, :], axis=0))\n",
    "    return sumY\n",
    "def theta_e(q):\n",
    "    return np.sum(q, axis=0)\n",
    "\n",
    "# Create some training end points by solving the forward kinematics equation\n",
    "numPoints = 30\n",
    "# Specifications from paper\n",
    "q1 = np.linspace(0, np.pi, num=numPoints)\n",
    "q2 = np.linspace(-np.pi, 0, num=numPoints)\n",
    "q3 = np.linspace(-np.pi/2, np.pi/2, num=numPoints)\n",
    "q1v, q2v, q3v = np.meshgrid(q1, q2, q3)\n",
    "q = np.vstack((q1v.flatten(), q2v.flatten(), q3v.flatten()))\n",
    "\n",
    "x = x_e(q)\n",
    "y = y_e(q)\n",
    "theta = theta_e(q)"
   ]
  },
  {
   "source": [
    "The code below partially shows the setup for the Neural Network, since the network requires linear activations at the output layer but tanh activations in the hidden layers we did not use the library provided with this class. Our neural network differs from the one proposed by Duka with an additional hidden layer with 50 neurons. Nonetheless, the overall structure is a MLP, Multilayer perceptron with tanh activations for hidden neurons and a linear activation at the output layer.\n",
    "\n",
    "The inputs are $\\{X_E, Y_E, \\Theta_E\\}$, which describe the Cartesian position and orientation of the end-effector.\n",
    "\n",
    "The outputs are $\\{q_1, q_2, q_3\\}$, the joint angles shown in the figure above.\n",
    "\n",
    "![Duka Neural Network](nn.png)"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {
    "executionInfo": {
     "elapsed": 715,
     "status": "ok",
     "timestamp": 1606506995046,
     "user": {
      "displayName": "Leo Ling",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhkCy_59QRiX50I6a8rcexp3HHvsDFtpk9mAxws=s64",
      "userId": "05198425524455268493"
     },
     "user_tz": 360
    },
    "id": "aFXRICXLRzN9"
   },
   "outputs": [],
   "source": [
    "# a feature_transforms function for computing\n",
    "# U_L L layer perceptron units efficiently\n",
    "def feature_transforms(a, w, activation):    \n",
    "    for W in w:\n",
    "        a = W[0] + np.dot(a.T, W[1:])\n",
    "        a = activation(a).T\n",
    "    return a\n",
    "# an implementation of our model employing a nonlinear feature transformation\n",
    "def model(x,w, activation):    \n",
    "    # feature transformation \n",
    "    f = feature_transforms(x,w[0], activation)\n",
    "    # compute linear combination and return\n",
    "    a = w[1][0] + np.dot(f.T,w[1][1:])\n",
    "    return a.T\n",
    "def nn(w, x):\n",
    "    return model(x, w, np.tanh)\n",
    "\n",
    "# In literature, it seems MSE is used, also including L2 normalization term\n",
    "def nnCost(w, x, y, epsilon=1e-3):\n",
    "    pred = nn(w, x)\n",
    "    presum = (y - pred)**2\n",
    "\n",
    "    return np.sum(presum) / (y.shape[1]) + epsilon * sumW / y.shape[1]\n",
    "\n",
    "layerSizes = [3, 100, 50, 3]\n",
    "w = initialize_network_weights(layerSizes, scale=1)\n",
    "numBatches = 50\n",
    "for ii in tqdm(range(numIter)):\n",
    "    alpha = .01/(1 + .01 * ii)\n",
    "    c[ii] = nnCost(w, xData[:, validIdx], yData[:, validIdx], epsilon=0)\n",
    "    \n",
    "    # Do mini batches, otherwise training set is too large\n",
    "    for jj in range(numBatches):\n",
    "        delta = grad(w, xData[:, miniBatchIdxs[jj, :]], yData[:, miniBatchIdxs[jj, :]])\n",
    "\n",
    "        # Since output is a list, we have to update things elementwise\n",
    "        for superlayerIdx in range(len(w)):\n",
    "            for layerIdx in range(len(w[superlayerIdx])):\n",
    "                w[superlayerIdx][layerIdx] -= alpha * delta[superlayerIdx][layerIdx]"
   ]
  },
  {
   "source": [
    "<img src=\"loss.png\" width=\"350\">"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "The network shows convergence with decreasing R2 loss as training progresses. Due to the large number of generated data points, we used 50 Minibatches per epoch with a total of 500 epochs. As well as decreasing training rate."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "# Calculate Z which is R2 error from expected\n",
    "pred = nn(w, xData) * np.pi\n",
    "# Check how close the proposed solution is to the original input xData\n",
    "xPred = x_e(pred)\n",
    "yPred = y_e(pred)\n",
    "thetaPred = theta_e(pred)\n",
    "\n",
    "inputPredNorm = np.vstack(((xPred-np.mean(x))/np.std(x),\n",
    "                        (yPred-np.mean(y))/np.std(y),\n",
    "                        thetaPred/np.pi))\n",
    "presum = np.abs(inputPredNorm - xData)\n",
    "\n",
    "# Plot meshgrid on countour plot\n",
    "for ii in range(3):\n",
    "    maxPresum = np.amax(presum[ii,:])\n",
    "    cs = ax.scatter(x, y, presum[ii, :],\n",
    "                    s=presum[ii,:]*5,\n",
    "                    c=colormap(presum[ii,:]/maxPresum))"
   ],
   "cell_type": "code",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "executionInfo": {
     "elapsed": 605656,
     "status": "ok",
     "timestamp": 1606507600042,
     "user": {
      "displayName": "Leo Ling",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhkCy_59QRiX50I6a8rcexp3HHvsDFtpk9mAxws=s64",
      "userId": "05198425524455268493"
     },
     "user_tz": 360
    },
    "id": "ow4nqOTZRzN_",
    "outputId": "801eaebc-d10e-46e8-fa7e-014f493dd2ca"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "source": [
    "<img src=\"err.png\" width=\"300\">"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "This figure shows the deviation from the inputs $\\{X_E, Y_E, \\Theta_E\\}$ for the generated outputs. Since we are able to trivially solve the forward kinematics equations, we can map the output in the joint space to the Cartesian space to measure the error of the proposed solution of the neural network. As we can see the Neural network performs more poorly at the edge of it's output range, but overall, there is little error in the proposed Cartesian output."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "#visualize manipulator\n",
    "import matplotlib.animation\n",
    "plt.rcParams[\"animation.html\"] = \"jshtml\"\n",
    "plt.rcParams['figure.dpi'] = 150  \n",
    "plt.ioff()\n",
    "theta = np.array([0,0,0])\n",
    "link1 = np.array([[0,0,0],[0, l1*np.cos(theta[0]), l1*np.sin(theta[0])]])\n",
    "link2 = np.array([link1[1],[0, link1[1,1] + l2*np.cos(theta[0]+theta[1]), link1[1,2]+l2*np.sin(theta[0]+theta[1])]])\n",
    "link3 = np.array([link2[1],[0, link2[1,1] + l3*np.cos(theta[0]+theta[1]+theta[2]), link2[1,2]+l3*np.sin(theta[0]+theta[1]+theta[2])]])\n",
    "fig = plt.figure()\n",
    "ax = plt.axes(projection='3d')\n",
    "line1 = ax.plot([link1[0,0], link1[1,0]], [link1[0,1], link1[1,1]], zs = [link1[0,2], link1[1,2]])[0]\n",
    "line2 = ax.plot([link2[0,0], link2[1,0]], [link2[0,1], link2[1,1]], zs = [link2[0,2], link2[1,2]])[0]\n",
    "line3 = ax.plot([link3[0,0], link3[1,0]], [link3[0,1], link3[1,1]], zs = [link3[0,2], link3[1,2]])[0]\n",
    "z_desired = np.linspace(1,3,100)\n",
    "\n",
    "def animate(i):\n",
    "    ax.cla()\n",
    "    theta = nn(w, np.array([0,1.5,z_desired[i]])) * np.pi\n",
    "    link1 = np.array([[0,0,0],[0, l1*np.cos(theta[0]), l1*np.sin(theta[0])]])\n",
    "    link2 = np.array([link1[1],[0, link1[1,1] + l2*np.cos(theta[0]+theta[1]), link1[1,2]+l2*np.sin(theta[0]+theta[1])]])\n",
    "    link3 = np.array([link2[1],[0, link2[1,1] + l3*np.cos(theta[0]+theta[1]+theta[2]), link2[1,2]+l3*np.sin(theta[0]+theta[1]+theta[2])]])\n",
    "    line1 = ax.plot([link1[0,0], link1[1,0]], [link1[0,1], link1[1,1]], zs = [link1[0,2], link1[1,2]], color = \"red\")[0]\n",
    "    line2 = ax.plot([link2[0,0], link2[1,0]], [link2[0,1], link2[1,1]], zs = [link2[0,2], link2[1,2]],color = \"yellow\")[0]\n",
    "    line3 = ax.plot([link3[0,0], link3[1,0]], [link3[0,1], link3[1,1]], zs = [link3[0,2], link3[1,2]], color = \"blue\")[0]\n",
    "    \n",
    "matplotlib.animation.FuncAnimation(fig, animate, frames=100)"
   ],
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": []
  },
  {
   "source": [
    "<img src=\"anim.png\" width=\"350\">"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "This code visualizes the link positions over the Cartesian space for an given set of joint angles."
   ],
   "cell_type": "markdown",
   "metadata": {}
  }
 ],
 "metadata": {
  "colab": {
   "name": "ik_nn.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7-final"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
