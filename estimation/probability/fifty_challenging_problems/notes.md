## Results for Problem 21 over 10M simulations

r, r, replace: P[Urn = A] = 0.637464175195425, P[Urn = B] = 0.362535824804575, N = 1741461, N_A = 1110119, N_B = 631342
r, r, without_replacement: P[Urn = A] = 0.5699820638526845, P[Urn = B] = 0.43001793614731554, N = 1463525, N_A = 834183, N_B = 629342


r, b, replace: P[Urn = A] = 0.47079810375094344, P[Urn = B] = 0.5292018962490566, N = 1179170, N_A = 555151, N_B = 624019
r, b, without_replacement: P[Urn = A] = 0.570476015850775, P[Urn = B] = 0.429523984149225, N = 1461632, N_A = 833826, N_B = 627806


b, r, replace: P[Urn = A] = 0.47021350415922686, P[Urn = B] = 0.5297864958407731, N = 1181710, N_A = 555656, N_B = 626054
b, r, without_replacement: P[Urn = A] = 0.570675285378394, P[Urn = B] = 0.4293247146216061, N = 1460079, N_A = 833231, N_B = 626848


b, b, replace: P[Urn = A] = 0.30870632849480917, P[Urn = B] = 0.6912936715051908, N = 895521, N_A = 276453, N_B = 619068
b, b, without_replacement: P[Urn = A] = 0.0, P[Urn = B] = 1.0, N = 616902, N_A = 0, N_B = 616902


## Joint probabilities

The unconditional joint probabilities under the regimes of REPLACE and W/O REPLACEMENT: 
REPLACE
r, r, A: 0.637464175195425 * (1741461 / (1741461 + 1179170 + 1181710 + 895521)) = 0.2221187779894683
r, b, A: 0.47079810375094344 * (1179170 / (1741461 + 1179170 + 1181710 + 895521)) = 0.11107769682316158
b, r, A: 0.47021350415922686 * (1181710 / (1741461 + 1179170 + 1181710 + 895521)) = 0.1111787400292365
b, b, A: 0.30870632849480917 * (895521 / (1741461 + 1179170 + 1181710 + 895521)) = 0.055314252374315254

r, r, B: 0.362535824804575 * (1741461 / (1741461 + 1179170 + 1181710 + 895521)) = 0.12632241546485276
r, b, B: 0.5292018962490566 * (1179170 / (1741461 + 1179170 + 1181710 + 895521)) = 0.12485718893398819
b, r, B: 0.5297864958407731 * (1181710 / (1741461 + 1179170 + 1181710 + 895521)) = 0.1252643630416366
b, b, B: 0.6912936715051908 * (895521 / (1741461 + 1179170 + 1181710 + 895521)) = 0.12386656534334081

WITHOUT REPLACEMENT
r, r, A: 0.5699820638526845 * (1463525 / (1463525 + 1461632 + 1460079 + 616902)) = 0.16676529116149935
r, b, A: 0.570476015850775 * (1461632 / (1463525 + 1461632 + 1460079 + 616902)) = 0.16669392167909003
b, r, A: 0.570675285378394 * (1460079 / (1463525 + 1461632 + 1460079 + 616902)) = 0.16657497254174117
b, b, A: 0.0 * (616902 / (1463525 + 1461632 + 1460079 + 616902)) = 0.0

r, r, B: 0.43001793614731554 * (1463525 / (1463525 + 1461632 + 1460079 + 616902)) = 0.1258146016763232
r, b, B: 0.429523984149225 * (1461632 / (1463525 + 1461632 + 1460079 + 616902)) = 0.12550753297889822
b, r, B: 0.4293247146216061 * (1460079 / (1463525 + 1461632 + 1460079 + 616902)) = 0.12531601487204072
b, b, B: 1.0 * (616902 / (1463525 + 1461632 + 1460079 + 616902)) = 0.12332766509040734

The book argues more or less as follows: if the first ball is red then you replace the ball since the highest 
probability you could achieve is through picking urn A if we see both red and urn B if we see red and black, as 
opposed to not replacing the ball if the first draw is a red and choosing urn A regardless of the second draw. That is
P(r, r, A | replace) + P(r, b, B | replace) > P(r, r, A | w/o replace) + P(r, b, A | w/o replace) since
0.2221187779894683 + 0.12485718893398819 = 0.3469759669234565 > 0.16676529116149935 + 0.16669392167909003 = 0.3334592128405894.

On the other hand, if we observe a black ball first, then do not replace since (and choose A if b/r and B if b/b over
B if b/r and B if b/b)
P(b, r, B | replace) + P(b, b, B | replace) < P(b, r, A | w/o replace) + P(b, b, B | w/o replace) since
0.1252643630416366 + 0.12386656534334081 = 0.2491309283849774 < 0.16657497254174117 + 0.12332766509040734 = 0.2899026376321485.


## Using a Bayesian approach

Does the MAP give the same answer?

    |       |w/o replacement            | replacement 
    |-------|---------------------------|-----------------------
    |r, r   |[0.5 0.5]                  |[0.57021277 0.42978723]
    |r, b   |[0.5 0.5]                  |[0.4011976 0.5988024] 
    |b, r   |[0.66445183 0.33554817]    |[0.57021277 0.42978723]
    |b, b   |[0, 1]                     |[0.4011976 0.5988024]      

Which table tells us that 
1. if the first ball is red the optimal action is (i) replace and (ii) if we observe r, r choose A whereas if r, b we 
should choose B since 0.57021277 > 0.5 and + 0.5988024 > 0.5 (we can compare probabilities across replacement categories
from the table above since the weights are equal (i.e., 1/2) by construction).
2. if the first ball is black the optimal action is (i) not replacing and (ii) if we observe b, r choose A whereas if
b, b choose B since 0.66445 > 0.57021277 and 1 > 0.5988024

## Discussion: 

Why was I perplexed to the extent I was? I had a hard time seeing that I could choose the action but then different
urns. For example, if the first ball is red and I want to compare the actions, I was perplexed by how to weight
0.57021277 and 0.4011976 vis-a-vis 0.5. But this isn't the relevant comparison because, even though I'm tying myself 
down by the action before the second ball I am not with the urn. So I could choose to replace and if I see r, r choose
urn A but urn B if observing r, b. In summation, I had difficulty parsing the estimated probabilities and understanding
what they implied vis-a-vis action and choice.