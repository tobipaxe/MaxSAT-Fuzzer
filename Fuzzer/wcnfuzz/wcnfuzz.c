/* Copyright (c) 2009 - 2010, Armin Biere, Johannes Kepler University. (cnfuzz)*/
/* Copyright (c) 2023 - 2024, Tobias Paxian, University of Freiburg. (modified version for wcnf's)*/

#include <assert.h>
#include <ctype.h>
#include <inttypes.h> // PRIu64
#include <stdint.h>   // uint64_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/times.h>
#include <unistd.h>

// for the implementation of the knuth "linear congruential generator"
// #define MOD 18446744073709551616

uint64_t _state = 0;

#define MAX 20
static int clause[MAX + 1];
uint64_t maxweight = 0;

// linear congruential generator
static uint64_t RAND() {

  _state *= 6364136223846793005ul;
  _state += 1442695040888963407ul;
  assert(_state);

  return _state;
}

void InitRNGT(uint64_t seed) { _state = seed; }

unsigned pick(unsigned l, unsigned r) {
  fflush(stdout);
  assert(l <= r);
  const unsigned delta = 1 + r - l;
  unsigned rnd = RAND() >> 32, scaled;
  const double fraction = rnd / 4294967296.0;
  scaled = delta * fraction;
  const unsigned res = scaled + l;
  assert(l <= res);
  assert(res <= r);
  return res;
}

uint64_t pick_uint64(uint64_t l, uint64_t r) {
  // printf("l %" PRIu64 ", r %" PRIu64 "\n", l, r);
  // fflush(stdout);
  assert(l <= r);
  const uint64_t delta = 1 + r - l;
  uint64_t rnd = RAND(), scaled;
  const __float128 fraction = rnd / (__float128)18446744073709551615ul;
  // const long double fraction = rnd / (long double)18446744073709551615ul;
  scaled = delta * fraction;
  const uint64_t res = scaled + l;
  assert(l <= res);
  assert(res <= r);
  return res;
}

static int numstr(const char *str) {
  const char *p;
  for (p = str; *p; p++)
    if (!isdigit(*p))
      return 0;
  return 1;
}

#define SIGN() ((pick(31, 32) == 32) ? -1 : 1)

int main(int argc, char **argv) {
  int i, j, k, l, m, n, o, p, sign, lit, layer, w, val, min, max, ospread,
      nsoft;
  int fp, eqs, ands, *arity, maxarity, lhs, rhs, tiny, small, xors3, xors4;
  int **unused, *nunused, allmin, allmax, qbf, wcnf, *quant;
  int seed, nlayers, *sclayers, **layers, *width, *low, *high, *clauses, *soft,
      *softorig;
  double *factor;
  uint64_t *weights, *weightsorig, sumweights, seedx, uBound;
  const char *options, *gitHash;
  char *mark, option[100], wcnfPrefix[21] = "";
  FILE *file;

  qbf = 0;
  wcnf = 0;
  nsoft = 0;
  tiny = 0;
  small = 0;
  seed = -1;
  seedx = 0;
  options = 0;
  gitHash = 0;
  uBound = 18446744073709551614ul;

  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      printf(
          "usage: cnfuzz [-h][-q][<seed>][<option-file>]\n"
          "\n"
          "  -h   print command line option help\n"
          "  -q   generate quantified CNF in QDIMACS format\n"
          "  -w   generate WCNF format for MaxSAT\n"
          "  -g   git hash of commit which was used to create this instance\n"
          "  -u   upper bound for sum of weights of MaxSAT instance\n"
          "\n"
          "  --tiny   between 5 and 10 variables per layer\n"
          "  --small  between 10 and 20 variables per layer\n"
          "\n"
          "If the seed is not specified it is calculated from the process id\n"
          "and the current system time (in seconds).\n"
          "\n"
          "The optional <option-file> lists integer options with their "
          "ranges,\n"
          "one option in the format '<opt> <lower> <upper> per line.\n"
          "Those options are fuzzed and embedded into the generated input\n"
          "in comments before the 'p cnf ...' header.\n");
      exit(0);
    }
    if (!strcmp(argv[i], "-q"))
      qbf = 1;
    else if ((!strcmp(argv[i], "--wcnf")) || (!strcmp(argv[i], "-w")))
      wcnf = 1;
    else if (!strcmp(argv[i], "--tiny"))
      tiny = 1;
    else if (!strcmp(argv[i], "--small"))
      small = 1;
    else if (!strcmp(argv[i], "-u")) {

      i++;
      if (i == argc) {
        printf("The option u has to be followed by a upper bound.\n");
        exit(1);
      }
      sscanf(argv[i], "%" PRIu64, &uBound);
    } else if (!strcmp(argv[i], "-g")) {

      i++;
      if (i == argc) {
        printf("The option g has to be followed by a hash.\n");
        exit(1);
      }
      gitHash = argv[i];
    } else if (numstr(argv[i])) {
      if (seedx > 0) {
        fprintf(stderr, "*** cnfuzz: multiple seeds\n");
        exit(1);
      }
      sscanf(argv[i], "%" PRIu64, &seedx);
    } else if (options) {
      fprintf(stderr, "*** cnfuzz: multiple option files\n");
      exit(1);
    } else
      options = argv[i];
  }

  if (seed < 0)
    seed = abs((int)(times(0) * getpid()) >> 1);

  srand(seed);
  if (seedx == 0)
    seedx = ((uint64_t)seed << 32) + rand();
  InitRNGT(seedx);

  printf("c seed %" PRIu64 "\n", seedx);
  if (gitHash != NULL)
    printf("c gitCommitHash %s\n", gitHash);

  fflush(stdout);

  if (qbf) {
    printf("c qbf\n");
    fp = pick(0, 3);
    if (fp)
      printf("c but forced to be propositional\n");
  }
  if (options) {
    file = fopen(options, "r");
    ospread = pick(0, 10);
    if ((allmin = pick(0, 1)))
      printf("c allmin\n");
    else if ((allmax = pick(0, 1)))
      printf("c allmax\n");
    printf("c %d ospread\n", ospread);
    if (!file) {
      fprintf(stderr, "*** cnfuzz: can not read '%s'\n", options);
      exit(1);
    }
    while (fscanf(file, "%s %d %d %d", option, &val, &min, &max) == 4) {
      if (!pick(0, ospread)) {
        if (allmin)
          val = min;
        else if (allmax)
          val = max;
        else
          val = pick(min, max);
      }
      printf("c --%s=%d\n", option, val);
    }
    fclose(file);
  }

  if (tiny)
    w = pick(5, 10);
  else if (small)
    w = pick(10, 20);
  else
    w = pick(10, 70);
  if (!wcnf)
    printf("c width %d\n", w);
  if (tiny || small)
    nlayers = pick(1, 3);
  else
    nlayers = pick(1, 20);
  int allaresoft = 0;
  if (wcnf) {
    allaresoft = pick(0, 5) ? 0 : 1;
    if (!tiny)
      nlayers = pick(1, 9);
    if (nlayers > 2 || allaresoft) {
      int ul = nlayers;
      ands = pick(0, ul) ? 0 : pick(0, 18 - nlayers);
      ul += ands ? nlayers - 1 : 1;
      eqs = pick(0, ul) ? 0 : pick(0, 18 - nlayers);
      ul += eqs ? nlayers - 1 : 1;
      xors3 = pick(0, ul) ? 0 : pick(0, 15 - nlayers);
      ul += xors3 ? nlayers - 1 : 1;
      xors4 = pick(0, ul) ? 0 : pick(0, 11 - nlayers);
      ul += xors4 ? nlayers - 1 : 1;
      w = pick(5, 45 - ul);
    }
    // else if (nlayers <= 2) {
    else {
      int ul = 1;
      ands = pick(0, ul + nlayers - 1) ? 0 : pick(0, 28 - ul);
      ul *= ands ? 2 : 1;
      eqs = pick(0, ul + nlayers - 1) ? 0 : pick(0, 32 - ul);
      ul *= eqs ? 2 : 1;
      xors3 = pick(0, ul + nlayers - 1) ? 0 : pick(0, 19 - ul);
      ul *= xors3 ? 2 : 1;
      xors4 = pick(0, ul + nlayers - 1) ? 0 : pick(0, 13 - ul);
      ul *= xors4 ? 2 : 1;
      w = pick(10, 45 - ul);
    }

    if (allaresoft && nlayers > 4)
      nlayers = pick(2, 4);
    printf("c width %d\n", w);

    maxweight = pick(1, 5);
    if (maxweight == 2)
      maxweight = pick(2, 32);
    else if (maxweight == 3)
      maxweight = pick(33, 256);
    else if (maxweight == 4)
      maxweight = pick(257, 65535);
    else if (maxweight == 5) {
      if (pick(0, 5))
        maxweight = pick(65536, 4294967295); // 2^32
      else
        maxweight = pick_uint64(4294967296ul, 9223372036854775807ul); // 2^63-1
      if (maxweight > uBound)
        maxweight = pick_uint64(1, uBound / 2);
    }
    printf("c max weight %" PRIu64 "\n", maxweight);
  } else {
    eqs = pick(0, 3) ? 0 : pick(0, 99);
    ands = pick(0, 2) ? 0 : pick(0, 99);
    xors3 = pick(0, 4) ? 0 : pick(0, 49);
    xors4 = pick(0, 5) ? 0 : pick(0, 29);
  }
  printf("c layers %d\n", nlayers);
  printf("c equalities %d\n", eqs);
  printf("c ands %d\n", ands);
  printf("c xors3 %d\n", xors3);
  printf("c xors4 %d\n", xors4);

  layers = calloc(nlayers, sizeof *layers);
  sclayers = calloc(nlayers, sizeof *sclayers);
  quant = calloc(nlayers, sizeof *quant);
  width = calloc(nlayers, sizeof *width);
  low = calloc(nlayers, sizeof *low);
  high = calloc(nlayers, sizeof *high);
  clauses = calloc(nlayers, sizeof *clauses);
  unused = calloc(nlayers, sizeof *unused);
  nunused = calloc(nlayers, sizeof *nunused);
  factor = calloc(nlayers, sizeof *factor);
  int allSoftAreUnit = pick(0, 4) ? 0 : 1;
  int numberOfSCLayers = 0;
  if (allaresoft && allSoftAreUnit)
    allSoftAreUnit = pick(0, 10) ? 0 : 1;
  for (i = 0; i < nlayers; i++) {
    if (wcnf) {
      if (allaresoft)
        sclayers[i] = 1;
      else
        sclayers[i] = pick(0, 1 + 3 * numberOfSCLayers) ? 0 : 1;
      if (sclayers[i]) {
        int ub = 8-numberOfSCLayers > 3 ? 8-numberOfSCLayers : 4;
        width[i] = pick(4, ub);
        numberOfSCLayers++;
      } else {
        width[i] = pick(5, w);
      }
    } else {
      sclayers[i] = 0;
      width[i] = pick(tiny ? 5 : 10, w);
    }

    quant[i] = (qbf && !fp) ? pick(0, 2) - 1 : 0;
    low[i] = i ? high[i - 1] + 1 : 1;
    high[i] = low[i] + width[i] - 1;
    m = width[i];
    if (i)
      m += width[i - 1];

    if (!wcnf)
      n = (pick(300, 450) * m) / 100;
    else if (sclayers[i]) {
      n = (pick(450, 700) * m) / 100;
    } else
      n = (pick(150, 300) * m) / 100;

    clauses[i] = n;
    factor[i] = n / (double)m;
    printf("c layer[%d] = [%d..%d] w=%d v=%d c=%d r=%.2f q=%d s=%d\n", i,
           low[i], high[i], width[i], m, n, factor[i], quant[i], sclayers[i]);

    nsoft += sclayers[i] * clauses[i];
    nunused[i] = 2 * (high[i] - low[i] + 1);
    unused[i] = calloc(nunused[i], sizeof *unused[i]);
    k = 0;
    for (j = low[i]; j <= high[i]; j++)
      for (sign = -1; sign <= 1; sign += 2)
        unused[i][k++] = sign * j;
    assert(k == nunused[i]);
  }

  arity = calloc(ands, sizeof *arity);
  maxarity = m / 2;
  if (maxarity >= MAX)
    maxarity = MAX - 1;
  for (i = 0; i < ands; i++)
    arity[i] = pick(2, maxarity);
  n = 0;
  for (i = 0; i < ands; i++)
    n += arity[i] + 1;
  m = high[nlayers - 1];
  mark = calloc(m + 1, 1);
  for (i = 0; i < nlayers; i++)
    n += clauses[i];
  n += 2 * eqs;
  n += 4 * xors3;
  n += 8 * xors4;

  int x = 0;
  if (!wcnf) {
    printf("p cnf %d %d\n", m, n);
  } else {
    soft = calloc(ands + eqs + xors3 + xors4, sizeof *soft);
    softorig = soft;

    for (i = 0; i < eqs + ands + xors3 + xors4; i++) {
      soft[i] = (pick(0, 10) || allaresoft) ? 1 : 0;
      if (soft[i])
        x++;
    }
    nsoft += x;
    n += x;

    weights = calloc(nsoft, sizeof *weights);
    weightsorig = weights;
    sumweights = 0;
    int tooBig = 0;

    if (nsoft > 0 && maxweight > uBound / nsoft)
      tooBig = 1;

    for (i = 0; i < nsoft; i++) {
      weights[i] = pick_uint64(1, maxweight);
      if (tooBig && (uBound - sumweights) < weights[i] + nsoft) {
        maxweight = (uBound - sumweights) / (nsoft - i);
        weights[i] = pick_uint64(1, maxweight);
        printf("c new max weight %" PRIu64 "\n", maxweight);

        if (nsoft == i + 1)
          weights[i] = uBound - sumweights;
      }
      sumweights += weights[i];
    }
    printf("c variables       %d\n", m + x);
    printf("c clauses         %d\n", n);
    printf("c hard clauses    %d\n", n - nsoft);
    printf("c soft clauses    %d\n", nsoft);
    printf("c sum of weights  %" PRIu64 "\n", sumweights);
    printf("c all cl soft     %d\n", allaresoft);
    printf("c all sc are unit %d\n", allSoftAreUnit);
  }
  if (qbf && !fp)
    for (i = 0; i < nlayers; i++) {
      if (!i && !quant[0])
        continue;
      fputc(quant[i] < 0 ? 'a' : 'e', stdout);
      for (j = low[i]; j <= high[i]; j++)
        printf(" %d", j);
      fputs(" 0\n", stdout);
    }
  for (i = 0; i < nlayers; i++) {
#ifdef DEBUG
    printf("\nLayer %d:\n", i);
#endif

    for (j = 0; j < clauses[i]; j++) {
      l = 3;
      while (l < MAX && pick(17, 19) != 17)
        l++;

      if (wcnf && sclayers[i])
        printf("%" PRIu64 " ", *weights++);
      else if (wcnf)
        printf("h ");

      for (k = 0; k < l; k++) {
        layer = i;
        while (layer && pick(3, 4) == 3)
          layer--;

        if (nunused[layer]) {
          o = nunused[layer] - 1;
          p = pick(0, o);
          lit = unused[layer][p];
          if (mark[abs(lit)])
            k--;
            continue;
          nunused[layer] = o;
          if (p != o)
            unused[layer][p] = unused[layer][o];
        } else {
          lit = pick(low[layer], high[layer]);
          if (mark[lit])
            continue;
          lit *= SIGN();
        }
        clause[k] = lit;
        mark[abs(lit)] = 1;
        printf("%d ", lit);
        if (wcnf && sclayers[i] && allSoftAreUnit)
          break;
      }
      printf("0\n");
      for (k = 0; k < l; k++)
        mark[abs(clause[k])] = 0;
    }
  }
  while (eqs-- > 0) {
#ifdef DEBUG
    printf("\neqs:\n");
#endif
    if (wcnf && *soft)
      snprintf(wcnfPrefix, sizeof(wcnfPrefix), "h %d ", ++m);
    else if (wcnf)
      strcpy(wcnfPrefix, "h ");

    i = pick(0, nlayers - 1);
    j = pick(0, nlayers - 1);
    k = pick(low[i], high[i]);
    l = pick(low[j], high[j]);
    if (k == l) {
      eqs++;
      continue;
    }
    k *= SIGN();
    l *= SIGN();

    printf("%s%d %d 0\n", wcnfPrefix, k, l);
    printf("%s%d %d 0\n", wcnfPrefix, -k, -l);
    if (wcnf && *soft++)
      printf("%" PRIu64 " %d 0\n", *weights++, -m);
  }
  while (--ands >= 0) {
#ifdef DEBUG
    printf("\nand:\n");
#endif

    if (wcnf && *soft)
      snprintf(wcnfPrefix, sizeof(wcnfPrefix), "h %d ", ++m);
    else if (wcnf)
      strcpy(wcnfPrefix, "h ");

    l = arity[ands];
    assert(l < MAX);
    i = pick(0, nlayers - 1);
    lhs = pick(low[i], high[i]);
    mark[lhs] = 1;
    lhs *= SIGN();
    clause[0] = lhs;

    printf("%s%d ", wcnfPrefix, lhs);
    for (k = 1; k <= l; k++) {
      j = pick(0, nlayers - 1);
      rhs = pick(low[j], high[j]);
      if (mark[rhs]) {
        k--;
        continue;
      }
      mark[rhs] = 1;
      rhs *= SIGN();
      clause[k] = rhs;
      printf("%d ", rhs);
    }
    printf("0\n");

    for (k = 1; k <= l; k++)
      printf("%s%d %d 0\n", wcnfPrefix, -clause[0], -clause[k]);

    for (k = 0; k <= l; k++)
      mark[abs(clause[k])] = 0;

    if (wcnf && *soft++)
      printf("%" PRIu64 " %d 0\n", *weights++, -m);
  }
  while (--xors3 >= 0) {
#ifdef DEBUG
    printf("\nXOR3:\n");
#endif

    if (wcnf && *soft)
      snprintf(wcnfPrefix, sizeof(wcnfPrefix), "h %d ", ++m);
    else if (wcnf)
      strcpy(wcnfPrefix, "h ");

    int lits[3];
    for (int k = 0; k < 3; k++) {
      j = pick(0, nlayers - 1);
      lits[k] = SIGN() * pick(low[j], high[j]);
    }
    printf("%s%d %d %d 0\n", wcnfPrefix, lits[0], lits[1], lits[2]);
    printf("%s%d %d %d 0\n", wcnfPrefix, lits[0], -lits[1], -lits[2]);
    printf("%s%d %d %d 0\n", wcnfPrefix, -lits[0], lits[1], -lits[2]);
    printf("%s%d %d %d 0\n", wcnfPrefix, -lits[0], -lits[1], lits[2]);

    if (wcnf && *soft++)
      printf("%" PRIu64 " %d 0\n", *weights++, -m);
  }
  while (--xors4 >= 0) {
#ifdef DEBUG
    printf("\nXOR4:\n");
#endif

    if (wcnf && *soft)
      snprintf(wcnfPrefix, sizeof(wcnfPrefix), "h %d ", ++m);
    else if (wcnf)
      strcpy(wcnfPrefix, "h ");

    int lits[4];
    for (int k = 0; k < 4; k++) {
      j = pick(0, nlayers - 1);
      lits[k] = SIGN() * pick(low[j], high[j]);
    }

    printf("%s%d %d %d %d 0\n", wcnfPrefix, lits[0], lits[1], lits[2], lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, lits[0], lits[1], -lits[2],
           -lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, lits[0], -lits[1], lits[2],
           -lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, lits[0], -lits[1], -lits[2],
           lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, -lits[0], lits[1], lits[2],
           -lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, -lits[0], lits[1], -lits[2],
           lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, -lits[0], -lits[1], lits[2],
           lits[3]);
    printf("%s%d %d %d %d 0\n", wcnfPrefix, -lits[0], -lits[1], -lits[2],
           -lits[3]);

    if (wcnf && xors4 != 0 && *soft++)
      printf("%" PRIu64 " %d 0\n", *weights++, -m);
    else if (wcnf && xors4 == 0 && *soft)
      printf("%" PRIu64 " %d 0\n", *weights, -m);
  }
  free(mark);
  free(clauses);
  free(arity);
  free(high);
  free(low);
  free(width);
  free(nunused);
  free(quant);
  for (i = 0; i < nlayers; i++)
    free(unused[i]);
  free(layers);
  free(sclayers);
  free(factor);
  free(unused);

  if (wcnf) {
    free(weightsorig);
    free(softorig);
  }
  return 0;
}
