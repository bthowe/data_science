r = .95
q1 = .99
q2 = .01
p1 = .009
p2 = .01
p3 = .45
p4 = .55

def part_d():
    print('Syndrome group effect: {0}, {1}'.format(p4 - p3, p4 > p3))  # the treatment effect if have the syndrome
    print('Non-syndrome group effect: {0}, {1}'.format(p2 - p1, p2 > p1))  # the treatment effect if you don't have the syndrome
    # notice that the probability you take the treatment conditional on having the syndrome, q2, is very small whereas the probability you take the treatment conditional on not having the syndrome is large
    # this means the muted treatment effect (that of those non-syndrome persons) is amplified while the amplified treatment effect (that of those syndrom persons) is muted by their respective propensities for taking the drug.

    pz1_x1 = lambda q_1, q_2, r: q_2 * r / (q_2 * r + q_1 * (1 - r))
    pz1_x0 = lambda q_1, q_2, r: (1 - q_2) * r / ((1 - q_2) * r + (1 - q_1) * (1 - r))

    print('Overall group effect (RD): {0}, {1}'.format(
        p4 * pz1_x1(q1, q2, r) + p2 * (1 - pz1_x1(q1, q2, r)) - p3 * pz1_x0(q1, q2, r) - p1 * (1 - pz1_x0(q1, q2, r)),
        p4 * pz1_x1(q1, q2, r) + p2 * (1 - pz1_x1(q1, q2, r)) > p3 * pz1_x0(q1, q2, r) - p1 * (1 - pz1_x0(q1, q2, r))
    )
    )

    print('True overall treatment effect (ACE): {0}, {1}'.format(p2 * (1-r) + p4 * r - p1 * (1 - r) - p3 * r, p2 * (1-r) + p4 * r > p1 * (1 - r) + p3 * r))
    # That is, the true overall effect should just be the weighted average group specific treatment effects, weighted by the likelihood of each group.

if __name__ == '__main__':
    part_d()
