#include <stdlib.h>
#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


// ----------------------------------------------------------------------
// data structures

unsigned max_variable;
unsigned num_clauses;

struct clause {
  size_t size;
  bool hard;
  int weight;
  int literals[];
  int * begin () { return literals; }
  int * end () { return literals + size; }
};

std::vector<clause*> clauses;

// state for reduction
unsigned granularity;
unsigned try_next;
std::vector<clause*> reduced_clauses;
int golden_exit;

// stats
unsigned tries;
unsigned success;
unsigned fail;

// ----------------------------------------------------------------------
// setup and cleanup

void initialize (void) {
  max_variable = 0;
  num_clauses = 0;
  granularity = 0;
  try_next = 0;
}

void reset (void) {
  // clean up state space.
  for (auto c : clauses)
    delete [] c;
  // delete [] reduced_file_name;
  clauses.clear ();
  max_variable = 0;
  num_clauses = 0;
}

// ----------------------------------------------------------------------
// parsing

static void add_new_clause (const std::vector<int>& literals, const int weight) {

  // First allocate clause and copy literals.
  size_t size = literals.size ();
  size_t bytes = sizeof (struct clause) + size * sizeof (int);
  clause * c = (clause*) new char [bytes];

  c->size = size;
  c->weight = weight;
  c->hard = false;
  if (c->weight < 0) c->hard = true;

  int * q = c->literals;
  for (auto lit : literals) {
    *q++ = lit;
    if ((unsigned) lit > max_variable) max_variable = lit;
  }
  clauses.push_back (c);
  reduced_clauses.push_back (c);
  num_clauses++;
}


static const char * file_name;
static const char * reduced_file_name;
static const char * executable;
static FILE * file;

static void parse (void)
{
  int ch;
  std::vector<int> clause;
  int next = 0, weight = 0;

  while ((ch = getc (file)) == 'c') {
    while ((ch = getc (file)) != '\n') {
      if (ch == EOF) {
        printf ("found no clauses in '%s'\n", file_name);
        fclose (file);
        return;
      }
    }
    if (fscanf (file, "%d", &next) == 1) break;
  }
  initialize ();                         // does nothing.

  while (true) {
    if (ch == 'h') {
      weight = -1;                 // negative weight means hard clause
      ch = 0;
    }
    else {
      if (next < 0 || next >= INT_MAX) {
        fprintf (stderr, "parser error in '%s': unexpected weight %d\n", file_name, next);
        exit (1);
      }
      weight = next;
    }
    while (true) {
      if (fscanf (file, "%d", &next) != 1) {
        break;
      }
      if (next == INT_MIN) {
        fprintf (stderr, "parser error in '%s': invalid nexteral %d\n", file_name, next);
        exit (1);
      }
      if (next) {
        clause.push_back (next);
        continue;
      }
      add_new_clause (clause, weight);
      clause.clear ();
      break;
    }
    if (fscanf (file, "%d", &next) != 1 && (ch = getc (file)) != 'h') break;
  }

  if (next) {
    fprintf (stderr, "parser error in '%s': missing terminating zero\n", file_name);
    exit (1);
  }

  fclose (file);
}

// ----------------------------------------------------------------------

static void print_clauses (void) {
  FILE * file = fopen (reduced_file_name, "w");
  if (!file) {
    fprintf (stderr, "could not open and read %s\n", file_name);
    exit (1);
  }
  for (auto c : clauses) {
    if (c->hard) fputs ("h ", file);
    else fprintf (file, "%d ", c->weight);
    for (auto lit : *c) {
      fprintf (file, "%d ", lit);
    }
    fputs ("0\n", file);
  }
  fclose (file);
}

// printing clauses to red-<wdimacs>
static void print_reduced (void) {
  FILE * file = fopen (reduced_file_name, "w");
  if (!file) {
    fprintf (stderr, "could not open and read %s\n", file_name);
    exit (1);
  }
  for (auto c : reduced_clauses) {
    if (c->hard) fputs ("h ", file);
    else fprintf (file, "%d ", c->weight);
    for (auto lit : *c) {
      fprintf (file, "%d ", lit);
    }
    fputs ("0\n", file);
  }
  fclose (file);
}

static int call_binary (void) {
  const char * sfx = "1>/dev/null 2>/dev/null </dev/null";
  char * mycmd = (char *)malloc (strlen (executable) + strlen (reduced_file_name) + strlen (sfx) + 5);
  int status, res;
  sprintf (mycmd, "./%s %s %s", executable, reduced_file_name, sfx);
  status = system (mycmd);
  free (mycmd);
  res = WEXITSTATUS (status);
  return res;
}

static bool update_state (bool reduce) {
  tries++;
  if (reduce) {
    success++;
    clauses.clear ();
    for (auto c: reduced_clauses) {
      clauses.push_back (c);
    }
    try_next = 0;
    granularity = granularity < clauses.size () ? granularity : clauses.size ();
  } else fail++;
  reduced_clauses.clear ();
  if (try_next == granularity) {
    granularity++;
    try_next = 0;
  }
  const unsigned size = clauses.size ();
  if (granularity > size) return false;
  const unsigned interval = size / granularity;
  assert (interval);
  const unsigned low = interval * try_next++;
  const unsigned high = interval * try_next;
  unsigned i;
  for (i = 0; i < low; i++) reduced_clauses.push_back (clauses[i]);
  for (i = high; i < size; i++) reduced_clauses.push_back (clauses[i]);
  return true;
}

// ---------------------------------------------------------------------

static void reduce_loop (void) {
  print_reduced ();                         // print state to reduced-file
  golden_exit = call_binary ();   // call binary on reduced-file
  int exit = golden_exit;
  //printf ("golden exit status: %d\n", golden_exit);
  while (true) {
    //printf ("exit status: %d\n", exit);
    if (!update_state (exit == golden_exit)) {
      print_clauses ();
      return;                               // cannot reduce anymore
    }
    print_reduced ();                       // print state to reduced-file
    exit = call_binary ();                  // call binary on reduced-file
  }
}

// ----------------------------------------------------------------------

// TODO: add stats
static void print_stats (void) {
  printf ("done reducing!\n");
  printf ("exit code: %d\n", golden_exit);
  printf ("tries: %d success: %d fail: %d\n", tries, success, fail);
  printf ("original size: %d reduced size: %ld\n", num_clauses, clauses.size ());
}

// ----------------------------------------------------------------------

const char * usage =
"usage: wmaxdd [ <option> ... ] [ <wdimacs> ] [ <reduced> ] [ <executable> ]\n"
"\n"
"where '<option>' can be one of the following\n"
"\n"
"  -h | --help        print this command line option summary\n"
"\n"
"'<wdimacs>' is the input file that produces the bug (WDIMACS format).\n"
"'<reduced>' is the reduced output file (WDIMACS format).\n"
"and '<executable>' is the buggy binary.\n"
;


int main (int argc, char** argv) {
  printf ("awesome WMAXSAT Delta Debugger\n");
  for (int i = 1; i != argc; i++) {
    const char * arg = argv[i];
    if (!strcmp (arg, "-h") || !strcmp (arg, "--help")) {
      fputs (usage, stdout);
      exit (0);
    }
    else if (arg[0] == '-') {
      fprintf (stderr, "invalid option '%s' (try '-h')\n", arg);
      exit (1);
    }
    else if (!file_name) {
      file_name = arg;
    }
    else if (!reduced_file_name) {
      reduced_file_name = arg;
    }
    else if (!executable) {
      executable = arg;
    }
    else {
      fprintf (stderr, "too many arguments '%s' and '%s' (try '-h')\n", file_name, arg);
      exit (1);
    }
  }
  if (argc < 4) {
    fprintf (stderr, "not enough arguments (try '-h')\n");
    exit (1);
  }
  assert (file_name);
  assert (reduced_file_name);
  assert (executable);
  if (!(file = fopen (file_name, "r"))) {
    fprintf (stderr, "could not open and read %s\n", file_name);
    exit (1);
  }
  printf ("reading from '%s'\n", file_name);
  parse ();
  
  // print_clauses ();
  printf ("reducing to %s\n", reduced_file_name);
  reduce_loop ();
  print_stats ();
  reset ();
}
