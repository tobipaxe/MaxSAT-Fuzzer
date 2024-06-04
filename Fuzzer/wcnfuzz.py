#!/usr/bin/env python3

import random
import argparse

wcnf_size = "normal"
upper_bound = 2**64 - 2
max_clause_length = 20
only_gbmo = False


def pick(a, b):
    if b <= a:
        return b
    return random.randint(a, b)


def SIGN():
    return random.choice([-1, 1])


def generate_wcnf():
    global wcnf_size, upper_bound, max_clause_length

    nlayers = 0
    eqs = 0
    ands = 0
    xors3 = 0
    xors4 = 0
    w = 0
    maxweight = 0
    uBound = upper_bound
    sclayers = []
    width = []
    low = []
    high = []
    clauses = []
    unused = []
    nunused = []
    factor = []
    allSoftAreUnit = 0 if pick(0, 4) == 0 else 1
    gbmo = False
    slight_weight_diffs = False
    numberOfSCLayers = 0
    nsoft = 0
    weights = []
    soft = []
    sumweights = 0
    mark = []
    wcnfPrefix = ""

    nlayers = pick(1, 10)

    allaresoft = 0 if pick(0, 5) else 1

    if nlayers > 2 or allaresoft:
        ul = min(nlayers, 8)
        ands = 0 if pick(0, ul) else pick(0, 18 - nlayers)
        ul += 1 if ands else nlayers - 1
        eqs = 0 if pick(0, ul) else pick(0, 18 - nlayers)
        ul += 1 if eqs else nlayers - 1
        xors3 = 0 if pick(0, ul) else pick(0, 15 - nlayers)
        ul += 1 if xors3 else nlayers - 1
        xors4 = 0 if pick(0, ul) else pick(0, 11 - nlayers)
        ul += 1 if xors4 else nlayers - 1
        w = pick(5, 45 - ul)
    else:
        ul = 1
        ands = 0 if pick(0, ul + nlayers - 1) else pick(0, 28 - ul)
        ul *= 2 if ands else 1
        eqs = 0 if pick(0, ul + nlayers - 1) else pick(0, 32 - ul)
        ul *= 2 if eqs else 1
        xors3 = 0 if pick(0, ul + nlayers - 1) else pick(0, 19 - ul)
        ul *= 2 if xors3 else 1
        xors4 = 0 if pick(0, ul + nlayers - 1) else pick(0, 13 - ul)
        ul *= 2 if xors4 else 1
        w = pick(10, 45 - ul)

    if allaresoft and nlayers > 4:
        nlayers = max(nlayers // 2, 1)
        
    if wcnf_size == "small":
        nlayers = max(nlayers // 2, 1)
    elif wcnf_size == "tiny":
        nlayers = max(nlayers // 3, 1)

    print(f"c width {w}")

    maxweight = pick(1, 5)

    if not pick(0, 6) and maxweight > 3:
        slight_weight_diffs = True
    if maxweight < 4 and uBound > 2**30:
        gbmo = True if not pick(0, 6) else False
    if gbmo or only_gbmo:
        gbmo = True
        maxweight = pick(1, 3)
    if maxweight == 2:
        maxweight = pick(2, 32)
    elif maxweight == 3:
        maxweight = pick(33, 256)
    elif maxweight == 4:
        maxweight = pick(257, 65535)
    elif maxweight == 5:
        if pick(0, 10):
            maxweight = pick(65536, 4294967295)  # 2^32
            nlayers = max(nlayers // 2, 1)
            # w = max(w // 2, 2)
        else:
            maxweight = pick(4294967296, 9223372036854775807)  # 2^63-1
            nlayers = max(nlayers // 3, 1)
            # w = max(w // 3, 2)
    if maxweight > uBound:
        maxweight = pick(1, uBound / 2)

    for i in range(nlayers):
        if allaresoft:
            sclayers.append(1)
        elif not numberOfSCLayers and nlayers - i <= 1 and ands + eqs + xors3 + xors4 < 3:
            sclayers.append(1 if pick(0, 8) else 0)
        else:
            sclayers.append(0 if pick(0, 1 + 3 * numberOfSCLayers) else 1)
        if sclayers[i]:
            ub = 8 - numberOfSCLayers if 8 - numberOfSCLayers > 3 else 4
            width.append(pick(4, ub))
            numberOfSCLayers += 1
        else:
            width.append(pick(5, w))

        low.append(high[i - 1] + 1 if i else 1)
        high.append(low[i] + width[i] - 1)
        m = width[i]
        if i:
            m += width[i - 1]

        if sclayers[i]:
            n = (pick(450, 700) * m) // 100
        else:
            n = (pick(100, 300) * m) // 100

        clauses.append(n)
        factor.append(n / m)

        print(
            f"c layer[ {i:<2}] = [ {low[i]:<3}.. {high[i]:<3}]  w={width[i]:<3} v={m:<3} c={n:<3} r={round(factor[i],2):<5} s={sclayers[i]}"
        )

        nsoft += sclayers[i] * clauses[i]
        # positive + negative
        nunused.append(2 * (high[i] - low[i] + 1))
        unused.append([])
        for j in range(low[i], high[i] + 1):
            for sign in [-1, 1]:
                unused[i].append(sign * j)
        assert len(unused[i]) == nunused[i]

    arity = []
    maxarity = max(m // 2, 2)
    if maxarity >= max_clause_length:
        maxarity = max_clause_length - 1
    for i in range(ands):
        arity.append(pick(2, maxarity))

    n = 0
    for i in range(ands):
        n += arity[i] + 1

    m = high[nlayers - 1]
    mark = [0] * (m + 1)
    for i in range(nlayers):
        n += clauses[i]

    n += 2 * eqs
    n += 4 * xors3
    n += 8 * xors4

    x = 0
    for i in range(eqs + ands + xors3 + xors4):
        soft.append(1 if pick(0, 10) or allaresoft else 0)
        if soft[i]:
            x += 1

    nsoft += x
    n += x

    weights = []
    tooBig = 0

    if nsoft > 0 and maxweight > uBound // nsoft:
        tooBig = 1
        gbmo = False
        slight_weight_diffs = False

    if slight_weight_diffs:
        if not pick(16, 18) == 16:
            max_weight_diffs = pick(1, (maxweight // 100) + 1)
            assert max_weight_diffs > 0
        else:
            weight_diffs = pick(1, 1000)
            max_weight_diffs = (
                weight_diffs if maxweight > weight_diffs else maxweight // 100
            )
            max_weight_diffs += 1
            assert max_weight_diffs > 0
        initial_weight = pick(100, (maxweight - max_weight_diffs) + 100)
        assert initial_weight > 0
    gbmo_partitions = []
    if gbmo and nsoft > 5:
        gbmo_partitions = [0]
        for i in range(len(sclayers)):
            if sclayers[i]:
                gbmo_partitions.append(gbmo_partitions[-1] + clauses[i])
        partitionProbability = pick(15, nsoft)
        gbmo_strategy = pick(4, 7) == 4
        multiplicator = 1
        gbmo_partitions.pop(0)
    else:
        gbmo = False

    sumweights = uBound + 1
    while sumweights > uBound:
        j = 0
        sumweights = 0
        for i in range(nsoft):
            if slight_weight_diffs:
                if pick(1, 50) == 1:
                    weights.append(pick(1, max_weight_diffs))
                    assert weights[-1] > 0
                else:
                    weights.append(initial_weight + pick(1, max_weight_diffs))
                    assert weights[-1] > 0
            elif gbmo:
                if not gbmo_strategy and len(gbmo_partitions) > j and gbmo_partitions[j] == i:
                    j += 1
                    multiplicator = sumweights
                elif i > 1 and pick(1, partitionProbability) == 1:
                    multiplicator = sumweights

                weights.append(pick(1, maxweight) * multiplicator)
                assert weights[-1] > 0
            else:
                weights.append(pick(1, maxweight))
                assert weights[-1] > 0
            if tooBig and (uBound - sumweights) < weights[i] + nsoft:
                maxweight = (uBound - sumweights) // (nsoft - i)
                weights[i] = pick(1, maxweight)
                assert weights[i] > 0

                if nsoft == i + 1:
                    weights[i] = uBound - sumweights
                    assert weights[i] > 0
            sumweights += weights[i]
        if sumweights > uBound:
            maxweight = pick(1, maxweight)
            gbmo_strategy = True
            multiplicator = 1
            weights.clear()


    # print(f"c softs: {nsoft} {len(weights)} {len(soft)}")
    print(f"c max weight           {maxweight}")
    print(f"c layers               {nlayers}")
    print(f"c equalities           {eqs}")
    print(f"c ands                 {ands}")
    print(f"c xors3                {xors3}")
    print(f"c xors4                {xors4}")
    print(f"c variables            {m + x}")
    print(f"c clauses              {n}")
    print(f"c hard clauses         {n - nsoft}")
    print(f"c soft clauses         {nsoft}")
    print(f"c sum of weights       {sumweights}")
    print(f"c all cl soft          {allaresoft}")
    print(f"c all sc are unit      {allSoftAreUnit}")
    print(f"c slight weight diffs  {slight_weight_diffs}")
    print(f"c GBMO forced          {gbmo}")
    if gbmo:
        print("c maxweight for each gbmo layer only")

    sizes = [0] * 21
    for i in range(nlayers):
        # if sclayers[i]:
        #     print("c soft clauses in layer", i, low[i], high[i], clauses[i])
        # else:
        #     print("c hard clauses in layer", i, low[i], high[i], clauses[i])
        for j in range(clauses[i]):
            clause_length = 3
            while clause_length < max_clause_length and pick(17, 19) == 17:
                clause_length += 1
            while clause_length <= 3 and clause_length > 1 and pick(11, 20) == 11:
                clause_length -= 1

            sizes[clause_length] += 1

            if sclayers[i]:
                print(weights.pop(0), end=" ")
            else:
                print("h", end=" ")

            clause = []

            hit = 0
            k = 0
            while k < clause_length:
                layer = i
                while layer and pick(3, 4) == 3:
                    layer -= 1

                if nunused[layer] and hit < 5:
                    o = nunused[layer] - 1
                    p = pick(0, o)
                    lit = unused[layer][p]
                    if mark[abs(lit)]:
                        hit += 1
                        continue
                    nunused[layer] = o
                    if p != o:
                        unused[layer][p] = unused[layer][o]
                else:
                    lit = pick(low[layer], high[layer])
                    if mark[lit] and hit < 10:
                        hit += 1
                        continue
                    hit = 0
                    lit *= SIGN()
                k += 1
                clause.append(lit)
                mark[abs(lit)] = 1
                print(f"{lit}", end=" ")
                if sclayers[i] and allSoftAreUnit:
                    break
            print("0")
            for variable in clause:
                mark[abs(variable)] = 0

    # if eqs > 0:
    #     print("c equalities", eqs)
    while eqs > 0:
        if soft.pop(0):
            m += 1
            wcnfPrefix = f"h {m}"
        else:
            wcnfPrefix = "h"

        k = l = 0
        while k == l:
            i = pick(0, nlayers - 1)
            j = pick(0, nlayers - 1)
            k = pick(low[i], high[i])
            l = pick(low[j], high[j])

        k *= SIGN()
        l *= SIGN()

        print(f"{wcnfPrefix} {k} {l} 0")
        print(f"{wcnfPrefix} {-k} {-l} 0")
        if wcnfPrefix != "h":
            print(f"{weights.pop(0)} {-m} 0")
        eqs -= 1

    # if ands > 0:
    #     print("c ands")
    while ands > 0:
        clause.clear()
        if soft.pop(0):
            m += 1
            wcnfPrefix = f"h {m}"
        else:
            wcnfPrefix = "h"

        l = arity[ands - 1]
        assert l < max_clause_length
        i = pick(0, nlayers - 1)
        lhs = pick(low[i], high[i])
        mark[lhs] = 1
        lhs *= SIGN()
        clause.append(lhs)

        print(f"{wcnfPrefix} {lhs}", end=" ")
        k = 1
        while k <= l:
            j = pick(0, nlayers - 1)
            rhs = pick(low[j], high[j])
            if mark[rhs]:
                continue
            k += 1
            mark[rhs] = 1
            rhs *= SIGN()
            clause.append(rhs)
            print(rhs, end=" ")
        print("0")
        for k in range(1, l + 1):
            print(f"{wcnfPrefix} {-clause[0]} {-clause[k]} 0")

        for k in range(l + 1):
            mark[abs(clause[k])] = 0

        if wcnfPrefix != "h":
            print(f"{weights.pop(0)} {-m} 0")

        ands -= 1

    # if xors3 > 0:
    #     print("c xors3")
    while xors3 > 0:
        if soft.pop(0):
            m += 1
            wcnfPrefix = f"h {m}"
        else:
            wcnfPrefix = "h"

        lits = []
        for k in range(3):
            j = pick(0, nlayers - 1)
            lits.append(SIGN() * pick(low[j], high[j]))

        print(f"{wcnfPrefix} {lits[0]} {lits[1]} {lits[2]} 0")
        print(f"{wcnfPrefix} {lits[0]} {-lits[1]} {-lits[2]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {lits[1]} {-lits[2]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {-lits[1]} {lits[2]} 0")

        if wcnfPrefix != "h":
            print(f"{weights.pop(0)} {-m} 0")

        xors3 -= 1

    # if xors4 > 0:
    #     print("c xors4")
    while xors4 > 0:
        if soft.pop(0):
            m += 1
            wcnfPrefix = f"h {m}"
        else:
            wcnfPrefix = "h"

        lits = []
        for k in range(4):
            j = pick(0, nlayers - 1)
            lits.append(SIGN() * pick(low[j], high[j]))

        print(f"{wcnfPrefix} {lits[0]} {lits[1]} {lits[2]} {lits[3]} 0")
        print(f"{wcnfPrefix} {lits[0]} {lits[1]} {-lits[2]} {-lits[3]} 0")
        print(f"{wcnfPrefix} {lits[0]} {-lits[1]} {lits[2]} {-lits[3]} 0")
        print(f"{wcnfPrefix} {lits[0]} {-lits[1]} {-lits[2]} {lits[3]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {lits[1]} {lits[2]} {-lits[3]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {lits[1]} {-lits[2]} {lits[3]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {-lits[1]} {lits[2]} {lits[3]} 0")
        print(f"{wcnfPrefix} {-lits[0]} {-lits[1]} {-lits[2]} {-lits[3]} 0")

        if xors4 != 0 and wcnfPrefix != "h":
            print(f"{weights.pop(0)} {-m} 0")
        elif xors4 == 0 and wcnfPrefix != "h":
            print(f"{weights[0]} {-m} 0")

        xors4 -= 1

    return 0


def initialize_seed():
    seed = random.getrandbits(64)
    return seed


def parse_arguments():
    global wcnf_size, upper_bound, only_gbmo

    parser = argparse.ArgumentParser(description="wcnfuzz a MaxSAT fuzzer")
    parser.add_argument(
        "--tiny", action="store_true", help="smallest option, less layers"
    )
    parser.add_argument(
        "--small", action="store_true", help="less layers"
    )
    parser.add_argument(
        "-u",
        "--upperBound",
        type=int,
        default=upper_bound,
        help="upper bound for sum of weights of MaxSAT instance",
    )
    parser.add_argument(
        "-g",
        "--gitHash",
        help="git hash of commit which was used to create this instance",
    )
    parser.add_argument(
        "--gbmo",
        action="store_true",
        help="only create multileve (GBMO) instances",
    )
    parser.add_argument("-s", "--seed", type=int, help="seed")

    args = parser.parse_args()

    if args.tiny:
        wcnf_size = "tiny"
    elif args.small:
        wcnf_size = "small"
    else:
        wcnf_size = "normal"

    gitHash = args.gitHash

    if args.gbmo:
        only_gbmo = args.gbmo

    if args.seed is None:
        seed = initialize_seed()
    else:
        seed = args.seed
    random.seed(seed)
    print("c seed", seed)

    if gitHash is not None:
        print("c gitCommitHash", gitHash)

    if args.upperBound > 0:
        upper_bound = args.upperBound


def main():
    parse_arguments()
    generate_wcnf()


if __name__ == "__main__":
    main()
