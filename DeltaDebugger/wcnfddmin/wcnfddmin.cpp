const char *usage =
    "usage: ddwcnf [ <option> ... ] [ <wdimacs> ]\n"
    "\n"
    "where '<option>' can be one of the following\n"
    "\n"
    "  -h   | --help                          print this command line option "
    "summary\n\n"
    "  -R   | --rounds                        number of rounds in which the "
    "given techniques are applied (standard 99). It breaks if the after the "
    "second round no reductions are possible.\n"
    "  -C   | --minCls                        minimize clause database\n"
    "  -V   | --minVars                       minimize variables\n"
    "  -L   | --minLits                       minimize literals\n"
    "  -W2O | --minWeights2One                minimize weights -> weight 1\n"
    "  -S2H | --minSoft2Hard                  minimize soft clauses -> hard "
    "clauses\n"
    "  -WBS | --weightBinarySearch            find best weight with binary"
    "search // attention this is cost intensive!\n"
    "  -SC  | --shuffleClauses                shuffle clauses (only in between "
    "rounds)\n"
    "  -SL  | --shuffleLiterals               shuffle literals (only in "
    "between "
    "rounds)\n"
    "  -RV  | --renameVariables               rename all variables if the "
    "variable name space is not contiguous.\n\n"
    "                                         These techniques are executed in "
    "the order "
    "clauses, variables, literals, weights2one, shuffle clauses, shuffle "
    "literals, rename variables\n"
    "                                         if no argument is given "
    "(minimization techniques), (shuffling+renaming), all are executed\n\n"
    "  -x   | --empty                         the empty instance is tested\n"
    "  -s   | --solver                        command line tool/solver to "
    "execute\n"
    "  -r   | --reducedWcnfName               name of the reduced wcnf\n"
    "  -k   | --keepFiles                     keep all files instead of only "
    "the smallest erroneus file\n"
    "  -p   | --percentOffBS <double p>       abortion criteria of "
    "BinarySearch changed (default p = 10), p==0 would be standard binary "
    "search. Abortion criteria: p*origweight+lowerBound>upperBound\n"
    "  -e   | --exit-codes <i j k>            valid exit codes of the solver\n"
    "  -i   | --inactiveReductionFactor <f>   f * active elements > elements "
    "database -- rewrite database.\n"
#ifdef LOGGING
    "  -l   | --logging                       print very verbose logging "
    "information using -l multiple times increases the verbosity\n"
#endif
    "\n"
    "and '<wdimacs>' is the input file in the new (2022) WDIMACS format.\n";
// NOT TESTED!! -- "The solver reads from '<stdin>' if no input file is
// specified.\n";

#include <sys/resource.h>
#include <sys/time.h>
#include <zlib.h> // parser

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
// #include <cstring> // fast parser
#include <algorithm> // count_if
#include <cassert>   // assertions
#include <cstring>
#include <csignal>
#include <filesystem>
#include <fstream>  // parseWcnfFile std::ifstream
#include <iomanip>  //std::setw
#include <iostream> //std::cout
#include <random>   // shuffle std::vector
#include <set>
#include <sstream> // parseWcnfFile std::istringstream
#include <string>
#include <unordered_map>
#include <vector>

// Solver data.
struct clause {
  size_t size;
  uint64_t weight;
  uint64_t previousWeight;
  uint64_t lowerBound;
  uint64_t upperBound;
  bool hard;
  bool active;

  // Flexible Array Member(FAM)
  int literals[];

  int *begin() { return literals; }
  int *end() { return literals + size; }
};

enum MODES {
  CLAUSES,
  LITERALS,
  VARIABLES,
  SOFT2HARD,
  WEIGHT2ONE,
  WEIGHTBINARYSEARCH // binary search
};
// counter on how often the reduction was successful. Automatically initialized
// with 0.
unsigned clausesModeCounter, variablesModeCounter, literalsModeCounter,
    weight2oneModeCounter, soft2hardModeCounter, weightbinarysearchModeCounter;

double percentInaccuracyForBinarySearch = 10;

MODES currentMode = MODES::CLAUSES;
std::string folder = "/tmp/"; // /tmp is in RAM
bool keepOnlySmallestFile = true;

// which pair of elements in activated to leave out only for the next call
std::pair<unsigned, unsigned> pair;
// a vector of the length of clauses, literals or variables
// which only tells us if the element is activated
std::vector<bool> activated;
// for VARIABLES reduction: variables<abs(literal), activated mapping>
std::unordered_map<int, int> variables;

const char *fileName;                 // given file name
std::string tmpWCNFFileName;          // temporary wcnf file
std::string lastTmpWCNFFileName = ""; // temporary wcnf file
std::string reducedWcnfName;          // name of the reduced wcnf
std::string solver;

auto rng = std::default_random_engine{};
// prefix for all fileNames during this call
std::string prefix;
std::vector<int> exitCodes = {0, 10, 20};
int expectedExitCode;
std::set<int> occuringExitCodes;
// keep original position of all clauses
bool positional = false;
std::vector<clause *> clauses;
// for testing renaming, shuffling, unweighting ... usw.
std::vector<clause *> clausesCopy;
// when to remove inactive Clauses -
// if the number of inactive clauses is that factor from the original database
unsigned inactiveReductionFactor;
// shuffle clauses if the size cannot be decreased for too long
unsigned fileCounter = 1;
// some statistic-variables
unsigned nbVars = 0;
// counts how often a empty file is checked. -- to check it only once!!
unsigned fileEmptyCounter = 0;
bool checkEmptyFile = false;
unsigned problemSizeBefore = 0;

double averageSolverCallTime = 0;

bool termination_flag = false;

// Signal handler function
void signalHandler(int signum) {
  // Handle the signal here
  std::cout << "c Interrupt signal (" << signum << ") received.\n";
  std::cout << "c Terminating program, this may take a while...\n";
  termination_flag = true;
}

void reset(void) {
  for (auto clause : clauses) {
    delete[] clause;
  }
  for (auto clause : clausesCopy) {
    delete[] clause;
  }
}

static void die(const char *fmt, ...) {
  fprintf(stderr, "c solver error: ");
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  reset();
  exit(1);
}

#ifdef LOGGING
unsigned logging = 0;

void debug(clause *clause, unsigned loglevel) {
  if (loglevel > logging)
    return;
  std::cout << "c DEBUG CLAUSE:  ";
  if (clause->hard)
    std::cout << "h ";
  else
    std::cout << clause->weight << " ";
  for (auto lit : *clause) {
    std::cout << lit << ", ";
  }
  std::cout << std::endl;
}

#define dout0 std::cout << "c DEBUG (" << __LINE__ << "): "
#define dout1                                                                  \
  if (logging > 0)                                                             \
  std::cout << "c DEBUG (" << __LINE__ << "): "
#define dout2                                                                  \
  if (logging > 1)                                                             \
  std::cout << "c DEBUG (" << __LINE__ << "): "
#define dout3                                                                  \
  if (logging > 2)                                                             \
  std::cout << "c DEBUG (" << __LINE__ << "): "

#else
#define dout0 0 && std::cout
#define dout1 0 && std::cout
#define dout2 0 && std::cout
#define dout3 0 && std::cout
#define debug(...)                                                             \
  do {                                                                         \
  } while (0)
#endif

//-------------------------------------------------------------------------------------------------
// A simple buffered character stream class:
// static const int buffer_size = 1048576;

class StreamBuffer {
  gzFile in;
  unsigned char buf[1048576];
  int pos;
  int size;

  void assureLookahead() {
    if (pos >= size) {
      pos = 0;
      size = gzread(in, buf, sizeof(buf));
    }
  }

public:
  explicit StreamBuffer(gzFile i) : in(i), pos(0), size(0) {
    assureLookahead();
  }

  int operator*() const { return (pos >= size) ? EOF : buf[pos]; }
  void operator++() {
    pos++;
    assureLookahead();
  }
  int position() const { return pos; }
};

//-------------------------------------------------------------------------------------------------
// End-of-file detection functions for StreamBuffer and char*:

static inline bool isEof(const StreamBuffer &in) { return *in == EOF; }
static inline bool isEof(const char *in) { return *in == '\0'; }

//-------------------------------------------------------------------------------------------------
// Generic parse functions parametrized over the input-stream type.

template <class B> static void skipWhitespace(B &in) {
  while ((*in >= 9 && *in <= 13) || *in == 32)
    ++in;
}

template <class B> static void skipLine(B &in) {
  for (;;) {
    if (isEof(in))
      return;
    if (*in == '\n') {
      ++in;
      return;
    }
    ++in;
  }
}

template <class B> static int parseInt(B &in) {
  int val = 0;
  bool neg = false;
  skipWhitespace(in);

  if (*in == '-')
    neg = true, ++in;
  else if (*in == '+')
    ++in;
  if (*in < '0' || *in > '9') {
    std::cerr << "PARSE ERROR!!! Unexpected char in parseInt: " << *in
              << std::endl;
    exit(6);
  }
  while (*in >= '0' && *in <= '9') {
    val = val * 10 + (*in - '0');
    ++in;
  }
  assert(val != 0);
  return neg ? -val : val;
}

template <class B> static uint64_t parseWeight(B &in) {
  uint64_t val = 0;
  skipWhitespace(in);
  if (*in == 'h') {
    ++in;
    if (*in != ' ') {
      std::cerr << "o PARSE ERROR! Unexpected char in parseWeight: " << *in
                << std::endl;
      exit(7);
    }
    return UINT64_MAX;
  }
  if (*in == '-') {
    std::cerr << "o PARSE ERROR! Unexpected negative weight" << std::endl;
    exit(8);
  } else if (*in == '+')
    ++in;
  if (*in < '0' || *in > '9') {
    std::cerr << "o PARSE ERROR! Unexpected char in parseWeight: " << *in
              << std::endl;
    exit(9);
  }
  while (*in >= '0' && *in <= '9')
    val = val * 10 + (*in - '0'), ++in;
  return val;
}

// String matching: consumes characters eagerly, but does not require random
// access iterator.
template <class B> static bool eagerMatch(B &in, const char *str) {
  for (; *str != '\0'; ++str, ++in)
    if (*str != *in)
      return false;
  return true;
}

void add_new_clause(const std::vector<int> &literals, uint64_t &weight) {
  size_t size = literals.size();
  size_t bytes = sizeof(struct clause) + size * sizeof(int);
  clause *cl = (clause *)new char[bytes];

  cl->size = size;
  cl->active = true;

  for (unsigned i = 0; i < literals.size(); i++)
    cl->literals[i] = literals[i];

  debug(cl, 3);
  cl->weight = weight;
  cl->previousWeight = 0;
  cl->lowerBound = 0;
  cl->upperBound = 0;
  cl->hard = (weight == UINT64_MAX);

  assert(clauses.size() <= (size_t)INT_MAX);
  clauses.push_back(cl);
}

clause *copyClause(clause *origClause) {
  size_t bytes = sizeof(struct clause) + origClause->size * sizeof(int);
  clause *cl = (clause *)new char[bytes];

  cl->size = origClause->size;
  cl->hard = origClause->hard;
  cl->weight = origClause->weight;
  // at the stage where copy clause is called, these values should be this
  // standard value!
  cl->active = true;
  cl->previousWeight = 0;
  cl->lowerBound = 0;
  cl->upperBound = 0;

  int *q = cl->literals;
  for (auto lit : *origClause)
    *q++ = lit;

  debug(cl, 4);
  assert(clauses.size() <= (size_t)INT_MAX);
  return cl;
}

void CopyClauses(const std::vector<clause *> &clFrom,
                 std::vector<clause *> &clTo) {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  for (auto clause : clTo)
    delete[] clause;
  clTo.clear();
  for (unsigned i = 0; i < clFrom.size(); i++)
    clTo.push_back(copyClause(clFrom[i]));
}

std::string SeparateFilename(std::string file, unsigned rnd) {
  dout3 << __PRETTY_FUNCTION__ << std::endl;
  auto position = file.find_last_of("/\\");
  file = file.substr(position + 1);
  position = file.find_last_of(".");
  if (position > 50)
    position = 50;
  file = file.substr(0, position);
  file = "ddmin_" + file + "_" + std::to_string(rnd);
  return file;
}

bool parseWCNF(std::string wcnfFile) {
  dout2 << __PRETTY_FUNCTION__ << std::endl;

  gzFile input_stream = gzopen(wcnfFile.c_str(), "rb");
  if (input_stream == nullptr ||
      !(std::filesystem::is_regular_file(wcnfFile))) {
    std::cout << "c ERROR! Could not open file: " << wcnfFile << std::endl;
    return false;
  }
  StreamBuffer in(input_stream);
  std::vector<int> clause;
  uint64_t weight;

  while (true) {
    if (isEof(in)) {
      break;
    } else if (*in == 'c' || *in == '\n' || isEof(in)) {
      skipLine(in);
      continue;
    }

    weight = parseWeight(in);
    clause.clear();
    while (true) {
      skipWhitespace(in);
      if (*in == '0') {
        skipLine(in);
        break;
      }
      clause.push_back(parseInt(in));
    }

    assert(clause.size() > 0);
    add_new_clause(clause, weight);
  }
  gzclose(input_stream);
  return true;
}

void dumpClause(clause *cl) {
  if (cl->hard)
    std::cout << "h ";
  else
    std::cout << cl->weight << " ";
  for (auto lit : *cl) {
    std::cout << lit << " ";
  }
  std::cout << "0" << std::endl;
}

void dumpWCNFClauses() {
  // unsigned inactiveClauses = 0;
  for (unsigned i = 0; i < activated.size(); i++) {
    if (!activated[i] || (pair.first <= i && pair.second >= i)) {
      continue;
    }
    dumpClause(clauses[i]);
  }
}

void dumpWCNFLiterals() {
  unsigned clIdx = 0;
  unsigned litIdx = 0;
  bool printClause = false;
  for (unsigned i = 0; i <= activated.size(); i++) {
    if (litIdx == clauses[clIdx]->size) {
      if (printClause) {
        std::cout << 0 << std::endl;
        printClause = false;
      }
      if (i == activated.size())
        break;
      ++clIdx;
      litIdx = 0;
    }
    if (activated[i] && (pair.first > i || pair.second < i)) {
      if (!printClause && clauses[clIdx]->hard)
        std::cout << "h ";
      else if (!printClause)
        std::cout << clauses[clIdx]->weight << " ";
      std::cout << clauses[clIdx]->literals[litIdx] << " ";
      printClause = true;
    }
    litIdx++;
  }
}

void dumpWCNFVariables() {
  bool printClause = false;
  for (auto clause : clauses) {
    for (auto literal : *clause) {
      if (!clause->active)
        continue;
      size_t it = variables.find(abs(literal))->second;
      // std::cout << "c literal " << literal << " iterator: " << it <<
      // std::endl;
      if (activated[it] && (pair.first > it || pair.second < it)) {
        if (!printClause && clause->hard)
          std::cout << "h ";
        else if (!printClause)
          std::cout << clause->weight << " ";
        std::cout << literal << " ";
        printClause = true;
      }
    }
    if (printClause) {
      std::cout << "0" << std::endl;
      printClause = false;
    }
  }
}

void dumpWCNFWeight2One() {
  unsigned index = 0;
  for (auto clause : clauses) {
    if (!clause->hard && clause->previousWeight != 0) {
      if (!activated[index] || (pair.first <= index && pair.second >= index)) {
        std::cout << "1 ";
      } else {
        std::cout << clause->weight << " ";
      }
      index++;
    } else if (!clause->hard) {
      assert(clause->weight == 1);
      std::cout << "1 ";
    } else {
      assert(clause->hard);
      std::cout << "h ";
    }
    for (auto lit : *clause) {
      std::cout << lit << " ";
    }
    std::cout << "0" << std::endl;
  }
}

void dumpWCNFSoft2Hard() {
  unsigned index = 0;
  for (auto clause : clauses) {
    if (!clause->hard && clause->previousWeight != 0) {
      if (!activated[index] || (pair.first <= index && pair.second >= index)) {
        std::cout << "h ";
      } else {
        std::cout << clause->weight << " ";
      }
      index++;
    } else if (!clause->hard) {
      assert(clause->weight > 0);
      std::cout << "1 ";
    } else {
      assert(clause->hard);
      std::cout << "h ";
    }
    for (auto lit : *clause) {
      std::cout << lit << " ";
    }
    std::cout << "0" << std::endl;
  }
}

void dumpWCNFWeightBinarySearch() {
  unsigned index = 0;
  for (auto clause : clauses) {
    if (!clause->hard && clause->previousWeight > 0) {
      if (pair.first <= index && pair.second >= index) {
        clause->previousWeight =
            (clause->lowerBound + clause->upperBound + 1) / 2;
        // if (clause->previousWeight == 0) {
        //   std::cout << "c lB: " << clause->lowerBound
        //             << " uB: " << clause->upperBound
        //             << " pW: " << clause->previousWeight
        //             << " weight: " << clause->weight << std::endl;
        //   std::cout << std::flush;
        // }
        // clause->previousWeight = clause->lowerBound +
        // (uint64_t)((clause->upperBound - clause->lowerBound) / 2 + 0.5)
        std::cout << clause->previousWeight << " ";
      } else {
        // if (clause->upperBound == 0) {
        //   std::cout << "c lB: " << clause->lowerBound
        //             << " uB: " << clause->upperBound
        //             << " pW: " << clause->previousWeight
        //             << " weight: " << clause->weight << std::endl;
        //   std::cout << std::flush;
        // }
        std::cout << clause->upperBound << " ";
      }
      index++;
    } else if (!clause->hard) {
      std::cout << clause->weight << " ";
    } else {
      assert(clause->hard);
      std::cout << "h ";
    }
    for (auto lit : *clause) {
      std::cout << lit << " ";
    }
    std::cout << "0" << std::endl;
  }
}

bool UpdateBounds(bool faultTriggered) {
  dout3 << __PRETTY_FUNCTION__ << std::endl;
  auto it = activated.begin();
  dout3 << "soft clauses before updating" << std::endl;
  for (auto clause : clauses) {
    if (!clause->hard && clause->previousWeight > 0)
      dout3 << *it << " & " << clause->weight << ": " << clause->lowerBound
            << " <= " << clause->upperBound << " PW: " << clause->previousWeight
            << std::endl;
  }
  if (!faultTriggered && pair.first != pair.second && activated.size() != 1) {
    return false;
  }
  unsigned index = 0;
  bool upperLowerDifferent = false;

  for (auto clause : clauses) {
    if (!clause->hard && clause->previousWeight > 0) {
      if (!activated[index]) {
        index++;
        continue;
      }

      if (pair.first <= index && pair.second >= index) {
        if (faultTriggered) {
          // update upper bound
          clause->upperBound = clause->previousWeight;
          assert(clause->upperBound > 0);
        } else {
          // update lower bound
          clause->lowerBound = clause->previousWeight;
          assert(clause->lowerBound < clause->upperBound);
        }

        uint64_t weightDistance =
            clause->weight * (percentInaccuracyForBinarySearch / 100);
        dout2 << "c                                 " << clause->lowerBound
              << " < " << clause->upperBound << "  clauselLB + WD "
              << (clause->lowerBound + 1) + weightDistance
              << "  WD: " << weightDistance << "  WEIGHT: " << clause->weight
              << " TRUE/FALSE: "
              << (clause->upperBound >
                  (clause->lowerBound + 1) + weightDistance)
              << std::endl;
        if (clause->upperBound > (clause->lowerBound + 1) + weightDistance) {
          // dout2 << "c                                  LB+1+WD>UB" <<
          // std::endl;
          upperLowerDifferent = true;
        } else {
          // std::cout << "c Lower +wd >= upper" << std::endl;
          activated[index] = false;
        }
      }
      index++;
    }
  }
  it = activated.begin();
  dout3 << "soft clauses after updating" << std::endl;
  for (auto clause : clauses) {
    if (!clause->hard && clause->upperBound > 0)
      dout3 << *it << " & " << clause->weight << ": " << clause->lowerBound
            << " <= " << clause->upperBound << " PW: " << clause->previousWeight
            << std::endl;
  }
  return upperLowerDifferent;
}

void dumpWCNF() {
  switch (currentMode) {
  case MODES::CLAUSES:
    dumpWCNFClauses();
    break;
  case MODES::LITERALS:
    dumpWCNFLiterals();
    break;
  case MODES::VARIABLES:
    dumpWCNFVariables();
    break;
  case MODES::WEIGHT2ONE:
    dumpWCNFWeight2One();
    break;
  case MODES::SOFT2HARD:
    dumpWCNFSoft2Hard();
    break;
  case MODES::WEIGHTBINARYSEARCH:
    dumpWCNFWeightBinarySearch();
    break;
  default:
    std::cerr << "Error: Mode not implemented!!" << std::endl;
    exit(10);
    break;
  }
}

void writeWCNF() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  dout2 << "Write WCNF file: " << tmpWCNFFileName << std::endl;
  // Save the rdbuf of cout.
  std::ofstream out(tmpWCNFFileName);
  auto *coutbuf = std::cout.rdbuf(); // save old buf
  std::cout.rdbuf(out.rdbuf());      // redirect std::cout to tmpWCNFFileName!
  dumpWCNF();
  std::cout.rdbuf(coutbuf);
}

bool removeFile(const std::string &filePath) {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  std::error_code ec; // For error handling without exceptions

  if (std::filesystem::remove(filePath, ec)) {
    // File removed successfully or didn't exist
    dout3 << "File removed successfully or didn't exist: " << filePath
          << std::endl;
    return true;
  } else {
    if (ec) { // If there was an error
      dout0 << "Error removing file " << filePath << ": " << ec.message()
            << std::endl;
      return false;
    }
    // No error means the file didn't exist, which is also considered successful
    dout3 << "No error means the file didn't exist, which is also considered "
             "successful"
          << std::endl;
    return true;
  }
}

bool testCurrentPairs() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  writeWCNF();
  dout0 << "Start application with WCNF file: " << tmpWCNFFileName << std::endl;
  if (std::filesystem::is_empty(tmpWCNFFileName)) {
    if (!checkEmptyFile) {
      dout0 << "Don't check empty files, continue!" << std::endl;
      return false;
    }
    dout0 << "WCNF file is empty" << std::endl;
    if (fileEmptyCounter >= 1) {
      dout1 << "Do not check empty file again!!" << std::endl;
      // if (!WIFEXITED(std::system(("rm " + tmpWCNFFileName).c_str())))
      //   dout0 << "problem with program execution" << std::endl;
      if (!removeFile(tmpWCNFFileName))
        std::cout << "c Error removing empty file: " << tmpWCNFFileName
                  << std::endl;
      return false;
    }
    fileEmptyCounter++;
  }
  ++fileCounter;
#ifdef LOGGING
  if (logging > 4)
    std::system(("cat " + tmpWCNFFileName).c_str());
#endif
  int rv = WEXITSTATUS(std::system(
      (solver + " " + tmpWCNFFileName + " > /dev/null 2>&1").c_str()));
  if (expectedExitCode == 0) {
    if (std::any_of(exitCodes.begin(), exitCodes.end(),
                    [&](int i) { return rv == i; })) {
      dout0 << "Valid return value: " << rv << std::endl;
      occuringExitCodes.insert(rv);
      // if (keepOnlySmallestFile &&
      //     !WIFEXITED(std::system(("rm " + tmpWCNFFileName).c_str())))
      //   dout0 << "problem with program execution" << std::endl;
      if (keepOnlySmallestFile && !removeFile(tmpWCNFFileName))
        std::cout << "c Error removing file: " << tmpWCNFFileName << std::endl;
      return false;
    }
    expectedExitCode = rv;
  }

  if (rv != expectedExitCode) {
    dout0 << "Valid return value: " << rv << std::endl;
    occuringExitCodes.insert(rv);
    // if (keepOnlySmallestFile &&
    //     !WIFEXITED(std::system(("rm " + tmpWCNFFileName).c_str())))
    //   dout0 << "problem with program execution" << std::endl;
    if (keepOnlySmallestFile && !removeFile(tmpWCNFFileName))
      std::cout << "c Error removing file: " << tmpWCNFFileName << std::endl;

    return false;
  }
  dout0 << "Invalid return value: " << rv << std::endl;
  // if (keepOnlySmallestFile && !lastTmpWCNFFileName.empty() &&
  //     !WIFEXITED(std::system(("rm " + lastTmpWCNFFileName).c_str())))
  //   dout0 << "problem with program execution" << std::endl;
  if (keepOnlySmallestFile && !lastTmpWCNFFileName.empty() &&
      !removeFile(lastTmpWCNFFileName))
    std::cout << "c Error removing file: " << tmpWCNFFileName << std::endl;

  lastTmpWCNFFileName = tmpWCNFFileName;
  return true;
}

bool runWholeProblem() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  MODES tmpMode = currentMode;
  currentMode = MODES::CLAUSES;
  // all (active) clauses
  pair.first = clauses.size();
  pair.second = clauses.size();
  activated = std::vector<bool>(clauses.size(), true);

  tmpWCNFFileName =
      folder + prefix + "_iteration_" +
      std::string(6 - std::min(6, (int)log10(fileCounter) + 1), '0') +
      std::to_string(fileCounter) + "_whole_remaining_problem" + ".wcnf";
  if (!testCurrentPairs()) {
    dout0 << "c All clauses together do not throw an error!" << std::endl;
    currentMode = tmpMode;
    return false;
  }
  currentMode = tmpMode;
  return true;
}

void RenameVars() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  for (auto clause : clauses) {
    for (unsigned i = 0; i < clause->size; ++i) {
      if (clause->literals[i] > 0)
        clause->literals[i] = variables.find(abs(clause->literals[i]))->second;
      else
        clause->literals[i] = -variables.find(abs(clause->literals[i]))->second;
    }
  }
}

void RemoveInactivePartsLiterals(unsigned &removed) {
  dout3 << __PRETTY_FUNCTION__ << std::endl;
  std::vector<bool>::iterator it = activated.begin();
  unsigned newSize = 0;

  for (auto clause : clauses) {
    unsigned counter = 0;
    std::vector<unsigned> newClause = {};
    for (size_t i = 0; i < clause->size; i++) {
      if (*it++) {
        clause->literals[counter] = clause->literals[i];
        counter++;
      }
    }

    if (counter == 0) {
      clause->active = false;
    } else {
      clause->size = counter;
      newSize += counter;
    }
  }
  removed = 0;

  activated = std::vector<bool>(newSize, true);
}

void RemoveInactivePartsVariables(unsigned &removed) {
  dout3 << __PRETTY_FUNCTION__ << std::endl;

  for (auto clause : clauses) {
    unsigned counter = 0;
    for (auto literal : *clause) {
      // size_t it = variables.find(abs(literal))->second;
      if (activated[variables.find(abs(literal))->second]) {
        clause->literals[counter] = literal;
        counter++;
      }
    }
    if (counter == 0) {
      clause->active = false;
    } else {
      clause->size = counter;
    }
  }
  removed = 0;
  variables.clear();
  int cnt = 0;
  for (auto clause : clauses) {
    for (auto literal : *clause) {
      auto found = variables.find(abs(literal));
      if (found == variables.end()) {
        variables.insert(std::pair<unsigned, unsigned>(abs(literal), cnt));
        cnt++;
        if (abs(literal) > (int)nbVars) {
          nbVars = abs(literal);
        }
      }
    }
  }
  activated = std::vector<bool>(variables.size(), true);
}

void RemoveInactiveParts(unsigned &removed, unsigned problemSize) {
  dout3 << "removed, inactiveClFa, problemsize: " << removed << ", "
        << inactiveReductionFactor << ", " << removed * inactiveReductionFactor
        << " < " << problemSize << std::endl;
  if (removed * inactiveReductionFactor < problemSize)
    return;

  dout2 << __PRETTY_FUNCTION__ << std::endl;
  removed = 0;

  switch (currentMode) {
  case MODES::CLAUSES:
    for (size_t i = 0; i < clauses.size(); ++i)
      if (!activated[i])
        clauses[i]->active = false;
    break;
  case MODES::LITERALS:
    RemoveInactivePartsLiterals(removed);
    break;
  case MODES::VARIABLES:
    RemoveInactivePartsVariables(removed);
    break;
  case MODES::WEIGHT2ONE: {
    unsigned actSize = activated.size();
    std::vector<bool>::iterator it = activated.begin();
    assert(actSize > 0);
    for (auto clause : clauses) {
      if (clause->previousWeight > 1) {
        if (!*it++) {
          clause->weight = 1;
          clause->previousWeight = 0;
          actSize--;
        }
      }
    }
    activated = std::vector<bool>(actSize, true);
  } break;
  case MODES::SOFT2HARD: {
    unsigned actSize = activated.size();
    std::vector<bool>::iterator it = activated.begin();
    assert(actSize > 0);
    for (auto clause : clauses) {
      if (!clause->hard && clause->previousWeight > 0) {
        if (!*it++) {
          clause->hard = true;
          clause->weight = UINT64_MAX;
          clause->previousWeight = 0;
          actSize--;
        }
      }
    }
    activated = std::vector<bool>(actSize, true);
  } break;
  case MODES::WEIGHTBINARYSEARCH: {
    dout2 << "Only change weights once at the end!" << std::endl;
  } break;
  default:
    std::cerr << "Error: Mode not implemented!!" << std::endl;
    exit(10);
    break;
  }

  clauses.erase(std::remove_if(begin(clauses), end(clauses),
                               [](clause *cl) {
                                 if (!cl->active) {
                                   delete[] cl;
                                   return true;
                                 }
                                 return false;
                               }),
                end(clauses));

  if (currentMode == MODES::CLAUSES) {
    activated = std::vector<bool>(clauses.size(), true);
    dout1 << "New Clauses size: " << clauses.size()
          << " activated size: " << activated.size() << std::endl;
  }
}

unsigned UpdateWeights() {

  // std::vector<bool>::iterator it = activated.begin();

  // unsigned redSize = 0;
  // std::count_if(clauses.begin(), clauses.end(), [](clause *cl) {
  //   uint64_t weightDistance =
  //     cl->weight * (percentInaccuracyForBinarySearch / 100);
  //   return !cl->hard && cl->weight > 1 && cl->weight > cl->upperBound +
  //   weightDistance;
  // });

  unsigned problemSize = 0;
  unsigned reduced = 0;
  unsigned reducedPlus = 0;
  unsigned reducedMinus = 0;

  dout0 << "ProblemSizeBeforeBefore: " << problemSizeBefore << std::endl;

  for (auto clause : clauses) {
    if (!clause->hard && clause->weight > 1) {
      problemSize++;
      // std::cout << *it << " & " << clause->weight << ": "
      //           << clause->lowerBound << " <= " << clause->upperBound
      //           << " PW: " << clause->previousWeight << std::endl;
      // clause->previousWeight = clause->upperBound;
      // if (clause->lowerBound + 1 == clause->upperBound) {

      uint64_t weightDistance =
          clause->weight * (percentInaccuracyForBinarySearch / 100);

      dout2 << "c                                 " << clause->lowerBound
            << " >= " << clause->upperBound << "  clauselLB + WD "
            << (clause->lowerBound + 1) + weightDistance
            << "  WD: " << weightDistance << "  WEIGHT: " << clause->weight
            << " TRUE/FALSE: "
            << (clause->upperBound <= (clause->lowerBound + 1) + weightDistance)
            << std::endl;

      if (clause->weight > clause->upperBound) {
        reduced++;
        if (clause->weight > clause->upperBound + weightDistance)
          reducedPlus++;
        if (clause->weight <= clause->upperBound + weightDistance) {
          // problemSizeBefore--;
          reducedMinus++;
        }
        clause->weight = clause->upperBound;
      }
    }
  }

  dout0 << " ProblemSizeBeforeAfter: " << problemSizeBefore << std::endl;
  dout0 << "ProblemSizeRecalculated: " << problemSize << std::endl;
  dout0 << "                Reduced: " << reduced << std::endl;
  dout0 << "            ReducedPlus: " << reducedPlus << std::endl;
  dout0 << "           ReducedMinus: " << reducedMinus << std::endl;

  for (auto clause : clauses) {
    clause->upperBound = clause->lowerBound = clause->previousWeight = 0;
  }
  return problemSizeBefore - reducedPlus;
}

std::pair<unsigned, unsigned>
CalculateNextBorders(unsigned size, unsigned arraySize, unsigned partIndex) {
  dout3 << __PRETTY_FUNCTION__ << std::endl;
  dout3 << "Current partIndex: " << partIndex << std::endl;
  if (arraySize == size && arraySize == 1)
    return std::make_pair(0, 1);

  unsigned numberArrays = size / arraySize;
  unsigned rest = size % arraySize;
  dout3 << "NumberArrays: " << numberArrays << " rest: " << rest
        << " size: " << size << "  clauses.size(): " << clauses.size()
        << "  arraySize: " << arraySize << std::endl;
  unsigned a = 0, b = 0;
  if (partIndex < rest) {
    a = partIndex * (arraySize + 1);
    b = a + arraySize;
  } else if (partIndex == rest) {
    a = rest * (arraySize + 1);
    b = a + arraySize - 1;
  } else {
    a = rest * (arraySize + 1) + (partIndex - rest) * arraySize;
    b = a + arraySize - 1;
  }
  dout3 << "a: " << a << " b: " << b << std::endl;
  // should never occur
  if (b >= size) {
    assert(a < size);
    b = size - 1;
  }

  return std::pair<unsigned, unsigned>(a, b);
}

bool runEmptyProblem() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  MODES tmpMode = currentMode;
  currentMode = MODES::CLAUSES;
  activated = std::vector<bool>(clauses.size(), false);
  tmpWCNFFileName =
      folder + prefix + "_empty_instance_count_iteration_" +
      std::string(6 - std::min(6, (int)log10(fileCounter) + 1), '0') +
      std::to_string(fileCounter) + ".wcnf";
  if (testCurrentPairs()) {
    // empty clause set throws still the error
    clauses = {};
    std::cout << "c Empty clause set throws error!" << std::endl;
    currentMode = tmpMode;
    return true;
  }
  currentMode = tmpMode;
  return false;
}

unsigned RenameVariabels() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  CopyClauses(clauses, clausesCopy);
  variables.clear();
  int counter = 1;
  for (auto clause : clauses) {
    for (auto literal : *clause) {
      auto found = variables.find(abs(literal));
      if (found == variables.end()) {
        variables.insert(std::pair<unsigned, unsigned>(abs(literal), counter));
        counter++;
        if (abs(literal) > (int)nbVars) {
          nbVars = abs(literal);
        }
      }
    }
  }
  if (nbVars > variables.size()) {
    RenameVars();
    if (!runWholeProblem()) {
      std::cout << "c Rename variables....: FAILED --> UNDO" << std::endl;

      CopyClauses(clausesCopy, clauses);
#ifdef LOGGING
      if (!runWholeProblem()) {
        std::cerr << "c Rename variables....: ERROR, UNDOING FAILED!"
                  << std::endl;
        reset();
        exit(1);
      }
#endif // LOGGING
    } else {
      std::cout << "c Rename variables....: SUCCESS" << std::endl;
    }
  } else {
    std::cout << "c Rename variables....: SKIP" << std::endl;
  }

  return variables.size();
}

void ShuffleClauses() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;

  CopyClauses(clauses, clausesCopy);
  std::shuffle(clauses.begin(), clauses.end(), rng);
  if (!runWholeProblem()) {
    std::cout << "c Shuffle clauses.....: FAILED --> UNDO" << std::endl;
    CopyClauses(clausesCopy, clauses);
#ifdef LOGGING
    if (!runWholeProblem()) {
      std::cerr << "c   Shuffle clauses.....: ERROR, UNDOING FAILED!"
                << std::endl;
      reset();
      exit(1);
    }
#endif // LOGGING
  } else {
    std::cout << "c Shuffle clauses.....: SUCCESS" << std::endl;
  }
}

void ShuffleLiteralsInClauses() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;

  for (auto clause : clauses) {
    std::shuffle(&clause->literals[0], &clause->literals[clause->size], rng);
  }
  if (!runWholeProblem()) {
    std::cout << "c Shuffle literals....: FAILED --> UNDO" << std::endl;
    CopyClauses(clausesCopy, clauses);
#ifdef LOGGING
    if (!runWholeProblem()) {
      std::cerr << "c   Shuffle literals.....: ERROR, UNDOING FAILED!"
                << std::endl;
      reset();
      exit(1);
    }
#endif // LOGGING
  } else {
    std::cout << "c Shuffle literals....: SUCCESS" << std::endl;
  }
}

unsigned CalcProblemSize() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;
  if (currentMode == MODES::CLAUSES) {
    return clauses.size();
  } else if (currentMode == MODES::LITERALS) {
    return std::accumulate(
        clauses.begin(), clauses.end(), 0,
        [](unsigned a, clause *b) { return b->active ? a + b->size : a; });
  } else if (currentMode == MODES::VARIABLES) {
    variables.clear();
    int counter = 0;
    for (auto clause : clauses) {
      for (auto literal : *clause) {
        auto found = variables.find(abs(literal));
        if (found == variables.end()) {
          variables.insert(
              std::pair<unsigned, unsigned>(abs(literal), counter));
          counter++;
          if (abs(literal) > (int)nbVars) {
            nbVars = abs(literal);
          }
        }
      }
    }
    return variables.size();
  } else if (currentMode == MODES::WEIGHT2ONE) {
    return std::count_if(clauses.begin(), clauses.end(), [](clause *cl) {
      return !cl->hard && cl->weight > 1;
    });
  } else if (currentMode == MODES::SOFT2HARD) {
    return std::count_if(clauses.begin(), clauses.end(),
                         [](clause *cl) { return !cl->hard; });
  } else if (currentMode == MODES::WEIGHTBINARYSEARCH) {
    return std::count_if(clauses.begin(), clauses.end(), [](clause *cl) {
      uint64_t weightDistance =
          cl->weight * (percentInaccuracyForBinarySearch / 100);
      return !cl->hard && cl->weight > 1 &&
             (cl->upperBound == 0 ||
              cl->upperBound > cl->lowerBound + 1 + weightDistance);
    });
  }

  return 0;
}

double ddmin() {
  dout2 << __PRETTY_FUNCTION__ << std::endl;

  unsigned problemSize = CalcProblemSize();
  problemSizeBefore = problemSize;
  if (problemSize == 0) {
    std::cout << "c   --> nothing to reduce as problem size is 0! --> CONTINUE "
              << std::endl;
    return 0;
  }
  dout3 << "ProblemSize: " << problemSize << std::endl;

  if (currentMode != MODES::WEIGHTBINARYSEARCH)
    std::cout << "c   wc time prediction..: "
              << averageSolverCallTime * 2 * problemSizeBefore << std::endl;
  if (currentMode == MODES::CLAUSES) {
    std::cout << "c   clauses before......: " << problemSizeBefore << std::endl;
  } else if (currentMode == MODES::VARIABLES) {
    std::cout << "c   variables before....: " << problemSizeBefore << std::endl;
  } else if (currentMode == MODES::LITERALS) {
    std::cout << "c   literals before.....: " << problemSizeBefore << std::endl;
  } else if (currentMode == MODES::WEIGHT2ONE) {
    std::cout << "c   weights != 1 before.: " << problemSizeBefore << std::endl;
  } else if (currentMode == MODES::SOFT2HARD) {
    std::cout << "c   soft clauses before.: " << problemSizeBefore << std::endl;
  } else if (currentMode == MODES::WEIGHTBINARYSEARCH) {
    std::cout << "c   weights > 1 before..: " << problemSizeBefore << std::endl;
  }

  if (currentMode == MODES::WEIGHT2ONE) {
    for (auto cl : clauses) {
      if (!cl->hard && cl->weight > 1 && cl->previousWeight == 0)
        cl->previousWeight = cl->weight;
    }
  } else if (currentMode == MODES::SOFT2HARD) {
    for (auto cl : clauses) {
      if (!cl->hard && cl->previousWeight == 0)
        cl->previousWeight = cl->weight;
    }
  } else if (currentMode == MODES::WEIGHTBINARYSEARCH) {
    unsigned solverCallPrediction = 0;
    for (auto cl : clauses) {
      if (!cl->hard && cl->weight > 1) {
        solverCallPrediction += log2(cl->weight);
        cl->upperBound = cl->weight;
        cl->previousWeight = cl->weight;
        cl->lowerBound = 0;
      } else {
        cl->upperBound = 0;
        cl->previousWeight = 0;
        cl->lowerBound = 0;
      }
    }
    std::cout << "c   #calls prediction...: " << solverCallPrediction
              << std::endl;
    std::cout << "c   wc time prediction..: "
              << solverCallPrediction * averageSolverCallTime << std::endl;
  }

  activated = std::vector<bool>(problemSize, true);
  assert(activated.size() == problemSize);

  unsigned arraySize = problemSize;
  unsigned tmpRemoved = 0;
  while (arraySize >= 1) {
    RemoveInactiveParts(tmpRemoved, problemSize);
    problemSize = CalcProblemSize();

    unsigned nbArrays = problemSize / arraySize;
    dout1 << "ArraySize: " << arraySize << std::endl;

    for (unsigned i = 1; i <= nbArrays; i++) {
      pair = CalculateNextBorders(problemSize, arraySize, i - 1);

      tmpWCNFFileName =
          folder + prefix + "_iter_" +
          std::string(6 - std::min(6, (int)log10(fileCounter) + 1), '0') +
          std::to_string(fileCounter) + "_sz-" + std::to_string(problemSize) +
          "_arrSz_" + std::to_string(pair.second - pair.first + 1) +
          "_arrIter_" + std::to_string(i) + ".wcnf";

      bool active = false;
      for (unsigned it = pair.first; it <= pair.second; it++) {
        if (activated[it]) {
          active = true;
          break;
        }
      }

      if (!active)
        continue;

      // std::cout << "activated: ";
      // for (auto act : activated) {
      //   std::cout << act << " ";
      // }
      // std::cout << std::endl;
      if (termination_flag) {
        dout0 << "Terminate minimization loop of current mode." << std::endl;
        break;
      }

      if (!testCurrentPairs()) {
        if (currentMode == MODES::WEIGHTBINARYSEARCH)
          if (UpdateBounds(false)) {
            // if only one element, update lower bound and do it again!!
            i--;
          }
        pair = std::pair<unsigned, unsigned>(1, 0);
        continue;
      }
      if (currentMode == MODES::WEIGHTBINARYSEARCH) {
        if (UpdateBounds(true)) {
          // update upper bound for all elements
          i--;
          pair = std::pair<unsigned, unsigned>(1, 0);
          continue;
        }
        tmpRemoved = std::count_if(activated.begin(), activated.end(),
                                   [](bool i) { return !i; });
        continue;
      }

      for (unsigned it = pair.first; it <= pair.second; it++)
        activated[it] = false;

      tmpRemoved += pair.second - pair.first + 1;
    }
    arraySize = arraySize == 1 ? 0 : arraySize / 2;
    // if (arraySize == 1) {
    //   arraySize = 0;
    // }
  }

  // here we want to remove all chunk anyway, so a
  RemoveInactiveParts(++tmpRemoved, 1);
  problemSize = CalcProblemSize();

  if (currentMode == MODES::CLAUSES) {
    std::cout << "c   clauses after.......: " << problemSize << std::endl;
    std::cout << "c   clauses removed.....: " << problemSizeBefore - problemSize
              << std::endl;
  } else if (currentMode == MODES::VARIABLES) {
    std::cout << "c   variables after.....: " << problemSize << std::endl;
    std::cout << "c   variables removed...: " << problemSizeBefore - problemSize
              << std::endl;
  } else if (currentMode == MODES::LITERALS) {
    std::cout << "c   literals after......: " << problemSize << std::endl;
    std::cout << "c   literals removed....: " << problemSizeBefore - problemSize
              << std::endl;
  } else if (currentMode == MODES::WEIGHT2ONE) {
    std::cout << "c   weights != 1 after..: " << problemSize << std::endl;
    std::cout << "c   weights set to 1....: " << problemSizeBefore - problemSize
              << std::endl;
  } else if (currentMode == MODES::SOFT2HARD) {
    std::cout << "c   soft clauses after..: " << problemSize << std::endl;
    std::cout << "c   soft set to hard cl.: " << problemSizeBefore - problemSize
              << std::endl;
  } else if (currentMode == MODES::WEIGHTBINARYSEARCH) {
    problemSize = UpdateWeights();
    std::cout << "c   unreduced weights be: " << problemSizeBefore << std::endl;
    std::cout << "c   unreduced weights af: " << problemSize << std::endl;
    std::cout << "c   reduced weights.....: " << problemSizeBefore - problemSize
              << std::endl;
  }

  return (double(problemSizeBefore - problemSize) / (double)problemSizeBefore) *
         100;
}

bool minimization(bool minCls, bool minVars, bool minLits, bool minSoftToHard,
                  bool minWeightsToOne, bool minWeightsByBinarySeach,
                  double timeStart, unsigned round) {
  bool returnValue = false;
  unsigned maxCounter = 2;
  double reductionAtLeast = 0.000000001;
  if (round > 2) {
    maxCounter = 1;
    reductionAtLeast = round - 2;
  }

  for (const auto mode :
       {MODES::CLAUSES, MODES::VARIABLES, MODES::LITERALS, MODES::WEIGHT2ONE,
        MODES::SOFT2HARD, MODES::WEIGHTBINARYSEARCH}) {
    if (termination_flag)
      break;
    struct rusage resources;
    getrusage(RUSAGE_CHILDREN, &resources);
    double timeBefore = (resources.ru_utime.tv_sec +
                         1.e-6 * (double)resources.ru_utime.tv_usec);
    currentMode = mode;
    unsigned fileCounterBefore = fileCounter;
    // bool current_rv = false;
    double current_rv = 0;
    switch (mode) {
    case MODES::CLAUSES:
      if (!minCls || clausesModeCounter >= maxCounter)
        continue;
      std::cout << "c MODE................: Minimize Clauses" << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        clausesModeCounter++;
      else
        clausesModeCounter = 0;
      break;
    case MODES::VARIABLES:
      if (!minVars || variablesModeCounter >= maxCounter)
        continue;
      std::cout << "c MODE................: Minimize Variables" << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        variablesModeCounter++;
      else
        variablesModeCounter = 0;
      break;
    case MODES::LITERALS:
      if (!minLits || literalsModeCounter >= maxCounter ||
          (minVars && minCls && round == 1))
        // do at first minVars && minCls twice, because minLits is the most cost
        // intensive!!
        continue;
      std::cout << "c MODE................: Minimize Literals" << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        literalsModeCounter++;
      else
        literalsModeCounter = 0;
      break;
    case MODES::WEIGHT2ONE:
      if (!minWeightsToOne || weight2oneModeCounter >= maxCounter)
        continue;
      std::cout << "c MODE................: Minimize by changing Weights to 1"
                << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        weight2oneModeCounter++;
      else
        weight2oneModeCounter = 0;
      break;
    case MODES::SOFT2HARD:
      if (!minSoftToHard || soft2hardModeCounter >= maxCounter)
        continue;
      std::cout << "c MODE................: Minimize by changing soft to "
                   "hard clauses"
                << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        soft2hardModeCounter++;
      else
        soft2hardModeCounter = 0;
      break;
    case MODES::WEIGHTBINARYSEARCH:
      if (!minWeightsByBinarySeach ||
          weightbinarysearchModeCounter >= maxCounter)
        continue;
      std::cout << "c MODE................: Minimize by finding optimal weight "
                   "with binary search (kind of)"
                << std::endl;
      current_rv = ddmin();
      if (current_rv < reductionAtLeast)
        weightbinarysearchModeCounter++;
      else
        weightbinarysearchModeCounter = 0;
      break;
    default:
      std::cout << "c MODE " << mode << " not implemented!" << std::endl;
      break;
    }

    if (current_rv > reductionAtLeast)
      returnValue = true;
    getrusage(RUSAGE_CHILDREN, &resources);
    double tmpTimeNow = (resources.ru_utime.tv_sec +
                         1.e-6 * (double)resources.ru_utime.tv_usec);
    std::cout << "c   solver calls........: " << fileCounter - fileCounterBefore
              << std::endl;
    std::cout << "c   time................: " << tmpTimeNow - timeBefore
              << std::endl;
    averageSolverCallTime = (tmpTimeNow - timeStart) / fileCounter;
#ifdef LOGGING
    runWholeProblem();
#endif
    dout1 << "current_rv: " << current_rv
          << "  reductionAtLeast: " << reductionAtLeast
          << "  returnValue: " << returnValue << std::endl;
  }
  return returnValue;
}

int main(int argc, char **argv) {
  dout2 << __PRETTY_FUNCTION__ << std::endl;

  signal(SIGTERM, signalHandler);
  signal(SIGINT, signalHandler);

  unsigned seed = abs((int)(time(NULL) * getpid()) >> 1);
  rng.seed(seed);
  unsigned rndPrefix = rng();
  double timeStart;
  struct rusage resources;
  getrusage(RUSAGE_CHILDREN, &resources);
  timeStart =
      resources.ru_utime.tv_sec + 1.e-6 * (double)resources.ru_utime.tv_usec;
  std::cout << "c Fancy WDIMACS Delta Debugger." << std::endl;
  std::cout << "c rnd number in file..: " << rndPrefix << std::endl;
  // rng.seed(0); // always same seed for all shuffling

  bool minCls = false;
  bool minVars = false;
  bool minLits = false;
  bool minWeightsToOne = false;
  bool minSoftToHard = false;
  bool minWeightsByBinarySeach = false;
  bool reducedAnything = false;
  unsigned numberRounds = 99;

  bool shuffleClauses = false;
  bool shuffleLiterals = false;
  bool renameVariables = false;

  inactiveReductionFactor = 5;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      fputs(usage, stdout);
      exit(0);
    }
    if (!strcmp(arg, "-s") || !strcmp(arg, "--solver")) {
      solver = argv[++i];
      std::replace(solver.begin(), solver.end(), '_', ' ');
      std::cout << "c solver..............: " << solver << std::endl;
    } else if (!strcmp(arg, "-R") || !strcmp(arg, "--rounds")) {
      numberRounds = strtoul(argv[++i], NULL, 0);
    } else if (!strcmp(arg, "-C") || !strcmp(arg, "--minCls")) {
      minCls = true;
    } else if (!strcmp(arg, "-V") || !strcmp(arg, "--minVars")) {
      minVars = true;
    } else if (!strcmp(arg, "-L") || !strcmp(arg, "--minLits")) {
      minLits = true;
    } else if (!strcmp(arg, "-W2O") || !strcmp(arg, "--minWeights2One")) {
      minWeightsToOne = true;
    } else if (!strcmp(arg, "-S2H") || !strcmp(arg, "--minSoft2Hard")) {
      minSoftToHard = true;
    } else if (!strcmp(arg, "-WBS") || !strcmp(arg, "--weightBinarySearch")) {
      minWeightsByBinarySeach = true;
    } else if (!strcmp(arg, "-SC") || !strcmp(arg, "--shuffleClauses")) {
      shuffleClauses = true;
    } else if (!strcmp(arg, "-SL") || !strcmp(arg, "--shuffleLiterals")) {
      shuffleLiterals = true;
    } else if (!strcmp(arg, "-RV") || !strcmp(arg, "--renameVariables")) {
      renameVariables = true;
    } else if (!strcmp(arg, "-p") || !strcmp(arg, "--percentOffBS")) {
      percentInaccuracyForBinarySearch = std::stod(argv[++i]);
    } else if (!strcmp(arg, "-k") || !strcmp(arg, "--keepFiles")) {
      keepOnlySmallestFile = false;
    } else if (!strcmp(arg, "-r") || !strcmp(arg, "--reducedWcnfName")) {
      reducedWcnfName = argv[++i];
    } else if (!strcmp(arg, "-x") || !strcmp(arg, "--empty")) {
      checkEmptyFile = true;
    } else if (!strcmp(arg, "-l") || !strcmp(arg, "--logging")) {
#ifdef LOGGING
      logging++;
#else
      die("compiled without logging code (use 'make --debug')");
      exit(1);
#endif
    } else if (!strcmp(arg, "-i") ||
               !strcmp(arg, "--inactiveReductionFactor")) {
      inactiveReductionFactor = strtoul(argv[++i], NULL, 0);
    } else if (!strcmp(arg, "-e") || !strcmp(arg, "--exit-codes")) {
      exitCodes.clear();
      while (++i < argc && isdigit(argv[i][0])) {
        exitCodes.push_back(strtoul(argv[i], NULL, 0));
      }
      --i;
      dout2 << exitCodes.size() << std::endl;
    } else if (arg[0] == '-')
      die("invalid option '%s' (try '-h')", arg);
    else if (fileName) {
      solver = solver + " " + fileName;
      fileName = arg;
      //      die("too many arguments '%s' and '%s' (try '-h')",
      //          fileName, arg);
    } else
      fileName = arg;
  }

  prefix = SeparateFilename(fileName, rndPrefix);
  std::cout << "c file prefix.........: " << prefix << std::endl;

  if (!minLits && !minVars && !minCls && !minWeightsToOne && !minSoftToHard &&
      !minWeightsByBinarySeach) {
    minLits = minVars = minCls = minWeightsToOne = minSoftToHard =
        minWeightsByBinarySeach = true;
  }
  if (!shuffleClauses && !shuffleLiterals && !renameVariables) {
    shuffleClauses = shuffleLiterals = renameVariables = true;
  }

  if (!fileName) {
    fileName = "<stdin>";
  }

  if (!parseWCNF(fileName)) {
    std::cerr << "ERROR --Parsing error!" << std::endl;
    reset();
    return 1;
  }

  getrusage(RUSAGE_CHILDREN, &resources);
  double tmpTimeBefore =
      (resources.ru_utime.tv_sec + 1.e-6 * (double)resources.ru_utime.tv_usec);
  if (!runWholeProblem()) {
    std::cout
        << "ERROR -- solver runs without throwing an error after reading the "
           "instance!"
        << std::endl;
    reset();
    return 2;
  }
  getrusage(RUSAGE_CHILDREN, &resources);
  double tmpTimeNow =
      (resources.ru_utime.tv_sec + 1.e-6 * (double)resources.ru_utime.tv_usec);
  averageSolverCallTime = (tmpTimeNow - tmpTimeBefore);
  std::cout << "c time 1. solver call.: " << averageSolverCallTime << std::endl;
  std::cout << "c invalid return code.: " << expectedExitCode << std::endl;

  // if (runEmptyProblem()) return 0;

  for (unsigned i = 1; i <= numberRounds; i++) {
    std::cout << "c ROUND " << i << std::endl;
    if (termination_flag) 
      break;
    bool rv =
        minimization(minCls, minVars, minLits, minWeightsToOne, minSoftToHard,
                     minWeightsByBinarySeach, timeStart, i);
    if (rv)
      reducedAnything = true;
    if (i > 1 && !rv)
      break;
    if (i != numberRounds) {
      if (shuffleClauses && clauses.size() > 1 && !termination_flag)
        ShuffleClauses();
      if (!termination_flag && shuffleLiterals &&
          (clauses.size() > 1 || (clauses.size() == 1 && clauses[0]->size > 1)))
        ShuffleLiteralsInClauses();
    }
    if (!termination_flag && renameVariables)
      RenameVariabels();
  }

  fileEmptyCounter = 0; // check even empty problem again!!
  if (!runWholeProblem()) {
    std::cerr
        << "ERROR -- solver runs without throwing an error after reduction!!!"
        << std::endl
        << std::endl;
    return 3;
  }

  if (reducedWcnfName == "")
    reducedWcnfName = "./red_" + prefix + ".wcnf";

  try {
    std::filesystem::rename(lastTmpWCNFFileName, reducedWcnfName);
    std::cout << "c moved successfully " + lastTmpWCNFFileName + " to " +
                     reducedWcnfName
              << std::endl;
  } catch (const std::filesystem::filesystem_error &e) {
    dout0 << "ERROR: problem with moving tmp file to given reduced path: "
          << e.what() << std::endl;
    return 4;
  }

  getrusage(RUSAGE_CHILDREN, &resources);
  tmpTimeNow =
      (resources.ru_utime.tv_sec + 1.e-6 * (double)resources.ru_utime.tv_usec);
  if (termination_flag) 
      std::cout << "c The Termination Flag is active, this means the reduction could've been better.: " << reducedAnything << std::endl;

  std::cout << "c Reduction successful: " << reducedAnything << std::endl;
  std::cout << "c return codes occured: ";
  for (auto i : occuringExitCodes)
    std::cout << i << ", ";
  std::cout << std::endl;
  std::cout << "c total solver calls..: " << fileCounter - 1 << std::endl;
  std::cout << "c av solver call time.: "
            << (tmpTimeNow - timeStart) / (fileCounter - 1) << std::endl;
  std::cout << "c time................: " << tmpTimeNow - timeStart
            << std::endl;
  reset();
  return 0;
}
