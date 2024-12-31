import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided text
mu_X = [4.670762791647576e-06, 3.857531282847049e-06, 3.5758180274569895e-06, 3.3711173728079302e-06, 
        3.0677304039272713e-06, 2.9202344649092993e-06, 3.6727967653860105e-06, 2.5274968720623292e-05, 
        0.0003885295591317117, 0.0018242523074150085, 0.004529042635113001, 0.007745007984340191, 
        12.947372436523438, 8.923138618469238, 7.671562194824219, 7.079471588134766, 7.120616436004639, 
        9.192477226257324, 14.278494834899902, 15.672234535217285, 10.222799301147461, 5.150974273681641, 
        1.998132348060608, -0.07186036556959152, -0.061112139374017715, -0.004212755709886551, 
        -0.002563177142292261, 0.004425741266459227, 0.01327630877494812, 0.019553236663341522, 
        -0.036383021622896194, -0.05858501419425011, -0.00841076672077179, -0.0034254470374435186, 
        0.024423819035291672, 0.272461861371994, 253.9636688232422, 222.851318359375, 216.8092498779297, 
        213.89529418945312, 211.03602600097656, 208.99658203125, 211.1063995361328, 217.28854370117188, 
        239.71754455566406, 261.45849609375, 272.70062255859375, 279.03289794921875]

sigma_X = [2.9060427664262534e-07, 4.052763244999369e-07, 5.055490532868134e-07, 5.795823199150618e-07, 
           5.788529051642399e-07, 5.809172876070079e-07, 1.1360400549165206e-06, 2.423883961455431e-05, 
           0.00041908351704478264, 0.0017827326664701104, 0.0036620653700083494, 0.00646115792915225, 
           35.91429901123047, 23.472213745117188, 20.224342346191406, 18.274974822998047, 15.742786407470703, 
           14.012439727783203, 14.048226356506348, 15.004902839660645, 10.846996307373047, 7.813650131225586, 
           6.6154255867004395, 5.065377712249756, 9.042277336120605, 6.528350353240967, 5.402632236480713, 
           4.709672927856445, 3.9010090827941895, 3.6641688346862793, 4.318202972412109, 5.071031093597412, 
           3.936681032180786, 2.901252031326294, 2.6451985836029053, 3.2065987586975098, 6.533928871154785, 
           11.613065719604492, 10.606907844543457, 9.817948341369629, 9.523937225341797, 11.174671173095703, 
           10.222972869873047, 6.731991291046143, 14.117526054382324, 15.375306129455566, 15.866228103637695, 
           20.93252944946289]

plt.figure(figsize=(15, 10), dpi=300)


# Plot for mu
plt.subplot(2, 1, 1)
plt.plot(mu_X, marker='o', linestyle='-', color='b', label='mu (Mean) - Variable X')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Mean (mu) of Variable X')
plt.grid(True)
plt.legend()

# Plot for sigma
plt.subplot(2, 1, 2)
plt.plot(sigma_X, marker='o', linestyle='-', color='r', label='sigma (Std Dev) - Variable X')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Standard Deviation (sigma) of Variable X')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save figure with high quality
plt.savefig('variable_x_statistics.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory