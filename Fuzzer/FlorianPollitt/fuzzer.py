import sys
from dataclasses import dataclass
import argparse
import random
import subprocess
from datetime import datetime
import os

def crunch_arguments():
  parser = argparse.ArgumentParser(
                    prog = 'fuzzer',
                    description = 'aWsome Cool New Fuzzer Fuzzer',
                    epilog = '''each instance is a tree of operators encoded with tseitin\n
                              by default one instance is created and printed to /tmp\n
                              if solver is specified the fuzzer will execute the created
                              instances\n
                              the deltadebugger will only be used if solver is specified\n''')
  parser.add_argument(
      "-s", "--solver", help="program to fuzz")
  parser.add_argument(
      "-d", "--deltadebugger", help="reduce instances")
  parser.add_argument(
      "-c", "--count", type=int, default=1,
      help="generate COUNT instances")
  parser.add_argument(
      "-p", "--path", default="/tmp",
      help="save instance in PATH (if stdout is specified it will print to stdout but this is incompatible with -s/-d)")
  parser.add_argument(
      "-e", "--errors", type=int, default=10,
      help="generate instances until ERRORS many different error showed up")
  parser.add_argument(
      "-a", "--allow", nargs="+", default=[0, 10, 20],
      help="exit codes that the solver is allowed to produce")
  parser.add_argument(
      "--seed", type=int,
      help="""seed results in one specific instance generated.
              seeds are also generated internally and have to be 20 digits long.""")
  parser.add_argument(
      "--max-depth", type=int, default=30,
      help="height of tree")
  parser.add_argument(
      "--max-width", type=int, default=15,
      help="width of tree")
  parser.add_argument(
      "--max-weight", type=int, default=50,
      help="")
  parser.add_argument(
      "--max-scale", type=int, default=666,
      help="")
  return parser.parse_args()


class Fuzzer:
  
  def __init__(self, args):
    self.instances = 0
    self.num_clauses = 0
    self.args = vars(args)
    self.random_degree = 0
    self.errors = set()
    self.seedsize = 10                        # actually twice as big
    self.current_vars = self.args["max_width"]
    self.wcnf = []
    if self.args["seed"] is None:
      self.update_seed()
    else:
      self.seed = self.args["seed"]
      assert len(str(self.seed)) == self.seedsize * 2
      random.seed(self.seed)
    
  def bug_name(self):
    return "bug-" + str(self.seed) + ".wcnf"

  def red_name(self):
    return "red-" + str(self.seed) + ".wcnf"

  def tmp_name(self):
    return self.args["path"] + "/fuzzer-" + str(self.seed) + ".wcnf"
  
  def get_next_rand(self, start, end):
    return random.randint(start, end)

  def update_seed(self):
    a = int.from_bytes(os.urandom(10), "big")
    a = int(str(a)[0:self.seedsize])
    assert len(str(a)) == self.seedsize
    b = int.from_bytes(os.urandom(10), "big")
    b = int(str(a)[0:self.seedsize])
    a = int(str(a) + str(b))
    random.seed(a)
    self.seed = a

  def operator_and(self, lits, top):
    # encode current_vars <-> and(lits)
    opneg = lambda x: -x
    left = list(map(opneg, lits)) + [top]
    opand = lambda x: [x, -top]
    right = list(map(opand, lits))
    right += [left]
    return right
  
  def operator_or(self, lits, top):
    # encode current_vars <-> or(lits)
    left = list(lits) + [-top]
    opor = lambda x: [-x, top]
    right = list(map(opor, list(lits)))
    right += [left]
    return right

  def operator_xor(self, lits, top):
    # encode current_vars <-> xor(lits) (a & -b) v (-a & b) <-> o
    # (a & -b) <-> o', (-a & b) <-> o'', (o' v o'') <-> o
    opneg = lambda x: -x
    cnf = []
    orlits = []
    for i, l in enumerate(lits):
      lits2 = lits.copy()
      lits2.remove(l)
      lits2 = [l] + list(map(opneg, lits2))
      o = top + i + 1
      cnf += self.operator_and(lits2, o) # not correct ?
      orlits += [o]
    cnf += self.operator_or(orlits, top)
    self.current_vars += len(lits)
    return cnf

  def operator_random(self, lits, top):
    cnf = []
    for l in range(self.get_next_rand(1, 3)):
      c = []
      for x in range(self.get_next_rand(1, 100)):
        c += [random.choice(list(lits))]
      cnf += [c]
    return cnf
    
  def add_final_constraint(self):
    if self.get_next_rand(0, 1):
      return
    cnf = [[self.current_vars * (-1) ** self.get_next_rand(0, 1)]]
    wcnf = self.generate_weights(cnf)
    self.extend_formula(wcnf)
  
  def generate_weights(self, cnf):
    all_hard = lambda x: ["h"] + x
    all_soft = lambda x: [self.get_next_rand(1, self.args["max_weight"])] + x
    fifty_fifty = lambda x: [["h", self.get_next_rand(1, self.args["max_weight"])][self.get_next_rand(0,1)]] + x
    weight = random.choice([all_hard, all_soft, fifty_fifty])
    return list(map(weight, cnf))

  def extend_formula(self, wcnf):
    self.wcnf += wcnf

  def scale_weights(self):
    if self.get_next_rand(0, 1):
      return
    scale = self.args["max_weight"]
    scalew = lambda x: [x[0] * self.get_next_rand(1, self.args["max_weight"])] + [i for i in x[1:]]
    hard = list(filter(lambda x: x[0] == "h", self.wcnf))
    soft = list(filter(lambda x: x[0] != "h", self.wcnf))
    self.wcnf = hard + list(map(scalew, soft))
    
  def scale_lits(self):
    if self.get_next_rand(0, 1):
      return
    scale = self.get_next_rand(1, self.args["max_scale"])
    scalelit = lambda x: [x[0]] + [i*scale for i in x[1:]]
    self.wcnf = list(map(scalelit, self.wcnf))

  def set_num_clauses(self):
    exp = self.get_next_rand(0, 2)
    num_clauses = self.get_next_rand(1, 9)
    for i in range(exp):
      num_clauses = num_clauses * 10 + self.get_next_rand(1, 9)
    
  def set_random_degree(self):
    self.random_degree = self.get_next_rand(0, 100)
    
  def set_num_hard_clauses(self):
    self.num_hard_clauses = self.get_next_rand(0, num_clauses)
    
  def create_instance2(self):
    self.set_num_clauses()
    self.set_random_degree()
    self.set_num_hard_clauses()
    

  def create_instance(self):
    self.instances += 1
    self.wcnf = []
    self.current_vars = self.args["max_width"]
    levels = self.get_next_rand(1, self.args["max_depth"])
    operators = [self.operator_and, self.operator_or,
                self.operator_xor, self.operator_random]
    for l in range(levels):
      width = self.get_next_rand(2, self.args["max_width"])
      self.current_vars += 1
      lits = set(range(1, self.current_vars))
      lits = set(map(lambda x: (-1) ** self.get_next_rand(0, 1) * x, lits))
      while len(lits) > width:
        l = random.choice(list(lits))
        lits.remove(l)
      operator = random.choice(operators)
      cnf = operator(lits, self.current_vars)
      wcnf = self.generate_weights(cnf)
      self.extend_formula(wcnf)
    self.add_final_constraint()
    self.scale_lits()
    self.scale_weights()

  def print_to_stdout(self):
    c = self.wcnf
    for i in c:
      sys.stdout.write(" ".join(list(map(str, i))) + " 0\n")
    return

  def write_to_file(self, file_name):
    f = open(file_name, "w")
    c = self.wcnf
    for i in c:
      f.write(" ".join(list(map(str, i))) + " 0\n")
    f.close()
    return

  
  def solve(self):
    if self.args["solver"] is None:
      return 0
    #options = " "
    #null = " 1>/dev/null 2>/dev/null </dev/null"
    o = ["1>/dev/null", "2>/dev/null", "</dev/null"]
    cmd = [self.args["solver"], self.tmp_name()]
    print(" ".join(cmd))
    p = subprocess.Popen(cmd)
    p.communicate()
    #print(p)
    return p.returncode
    #cmd = self.args["solver"] + options + self.tmp_name + null
    #subprocess.call(cmd, shell=True)

  def delta_debug(self, exit_code):
    if self.args["solver"] is None or self.args["deltadebugger"] is None:
      return 0
    self.write_to_file(self.bug_name())
    try:
    #cmd = [self.args["deltadebugger"], "-C", "-L", "-e", "0", "-r", self.red_name(), "-s", self.args["solver"], self.bug_name()]
      cmd = [self.args["deltadebugger"], self.bug_name(), self.red_name(), self.args["solver"]]
      p = subprocess.Popen(cmd)
      p.communicate()
    except:
      print("could not delta debug, smth went wrong!")
    

  def loop(self):
    while self.instances < self.args["count"] and len(self.errors) < self.args["errors"]:
      self.create_instance()
      if self.args["path"] == "stdout":
        self.print_to_stdout()
      else:
        self.write_to_file(self.tmp_name())
      exit_code = self.solve()
      if exit_code not in self.args["allow"] and exit_code is not None:
        self.errors.add(exit_code)
        self.delta_debug(exit_code)
      self.update_seed()
    
if __name__ == "__main__":
  f = Fuzzer(crunch_arguments())
  f.loop()
  
