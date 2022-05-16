"""6.009 Lab 9: Carlae Interpreter Part 2"""

from inspect import ArgSpec
import sys
sys.setrecursionlimit(10_000)

import doctest

# NO ADDITIONAL IMPORTS!


###########################
# Carlae-related Exceptions #
###########################


class CarlaeError(Exception):
    """
    A type of exception to be raised if there is an error with a Carlae
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class CarlaeSyntaxError(CarlaeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class CarlaeNameError(CarlaeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class CarlaeEvaluationError(CarlaeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    CarlaeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Carlae
                      expression
    """
    tokens = []
    separation = source.split("\n")
    to_process = []

    for line in separation:
        s = ""
        for char in line:
            if char == "#":
                break
            elif char == " ":
                if s != "" and s != " ":
                    to_process.append(s)

                s = ""
            else:
                s += char

        if s != "":
            to_process.append(s)

    def inner_tokenize(part):
        if len(part) > 1:
            if part[0] == "(":
                return ["("] + inner_tokenize(part[1:])

            elif part[0] == ")":
                return [")"] + inner_tokenize(part[1:])

            elif part[-1] == ")":
                return inner_tokenize(part[:-1]) + [")"]

            elif part[-1] == "(":
                return inner_tokenize(part[:-1]) + ["("]

            else:
                return [part]
        else:

            return [part]

    for part in to_process:
        tokens.extend(inner_tokenize(part))

    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def parse_expression(index, outer_call=True):
        token = tokens[index]
        value = number_or_symbol(token)
        if value == "(" and len(tokens) > 1:
            l = []
            next = index+1
            exp_value, new_index = parse_expression(next, False)
            while exp_value != ")":

                l.append(exp_value)
                try:
                    exp_value, new_index = parse_expression(new_index, False)
                except IndexError:
                    raise CarlaeSyntaxError

            try:
                if tokens[new_index] == ")" and new_index + 1 == len(tokens) and outer_call:
                    raise CarlaeSyntaxError
                else:
                    return (l, new_index)
            except IndexError:
                return (l, new_index)
        elif value == "(" and len(tokens) == 1:
            raise CarlaeSyntaxError

        elif value == ")" and outer_call:
            raise CarlaeSyntaxError
        elif value == ":=" and outer_call:
            raise CarlaeSyntaxError

        else:
            return (value, index+1)

    parsed_expression, next_index = parse_expression(0)

    return parsed_expression


######################
# Built-in Functions #
######################

def prod(list):
    prod = 1
    for i in list:
        prod = prod*i
    return prod


def div(list):
    division = list[0]/list[1]
    for i in range(2, len(list)):
        division = division/list[i]

    return division

def equal(list):
    i = 0

    while i < len(list)-1:
        if list[i] != list[i+1]:
            return False
        else:
            i +=1
    return True

def decreasing(list):
    i = 0

    while i <len(list) -1:
        if list[i]>list[i+1]:
            i +=2
        else:
            return False
    return True

def non_increasing(list):
    i = 0

    while i<len(list) - 1:
        if list[i]<list[i+1]:
            return False
            
        else:
            i +=2
    
    return True

def increasing(list):
    i = 0
    while i<len(list) - 1:
        if list[i]<list[i+1]:
            i +=2
            
        else:
            return False
    
    return True

def non_decreasing(list):
    i = 0
    while i<len(list) - 1:
        if list[i] > list[i+1]:
            return False
        else:
            i+=2
    return True

def no(args):
    if len(args) > 1 or len(args) == 0:
        raise CarlaeEvaluationError
    else:
        return not args[0]

def pair(list):
    if len(list) != 2:
        raise CarlaeEvaluationError
    else:
        return Pair(list[0],list[1])
    
def head(pair):
    if pair ==[] or len(pair)>1:
         raise CarlaeEvaluationError("Eroor in tail function")
    else:
        if type(pair[0]) != Pair :
            raise CarlaeEvaluationError("Error in head function")
        else:
            return pair[0].head

#how to make sure it does not block the linked list in the first call here: (tail (tail (tail (list 1 2 3 4))))
def tail(pair):
    if pair ==[] or len(pair)>1:
         raise CarlaeEvaluationError("Eroor in tail function")
    else:
        if type(pair[0]) != Pair :
            raise CarlaeEvaluationError("Eroor in tail function")
        else:
            return pair[0].tail

def make_list(args):
    if len(args) == 0:
        return []
    elif len(args) == 1:
        return Pair(args[0],[])
    else:
        return Pair(args[0],make_list(args[1:]))

def is_linked_list(object):
    
    def inner_check(pair):
        if pair == []:
            return "@t"
        else:
            if type(pair) != Pair:
                return "@f"
            else:
                return inner_check(pair.tail)
    
    if type(object[0]) == Pair:
        return inner_check(object[0])
    elif object[0] == []:
        return "@t"

    return "@f"

def length_linked_list(list):

    def get_length(l):
        if l == []:
            return 0
        elif l.tail == [] :
            return 1
        else:
            return 1 + get_length(l.tail)

    # print("list",list)
    if is_linked_list(list) == "@t":
        return get_length(list[0])
    
    raise CarlaeEvaluationError("Object is not a linked list, cannot get length")


def get_index(list):

    def inner(l,ind):

        if ind == 0 and l != []:
            return l.head
        elif l == []:
            raise CarlaeEvaluationError
        else:
            return inner(l.tail,ind-1)
    
    interest = list[0]
    index = list[-1]

    if is_linked_list([interest])== "@f" and index != 0:
        raise CarlaeEvaluationError("It is a pair, but asking for index that is not 0")

    else:
        return inner(interest,index)

def concat(args):

    if len(args) == 0:
        return []

    if is_linked_list([args[0]]) == "@t":

        if len(args) == 1:
            return args[0]

        elif args[0] == []:
            return concat(args[1:])

        else:
            args.insert(1,args[0].tail)
            return Pair(args[0].head,concat(args[1:]))
                
    else: 
        raise CarlaeEvaluationError("Passing in a non-list element to concat")

def map_func(args):
    func = args[0]

    def inner_map(l):
        if l == []:
            return l
        elif l.tail == []:
            res = func([l.head])
            return Pair(res,[])
        else:
           res = func([l.head])
           return Pair(res,inner_map(l.tail))
    
    if is_linked_list([args[1]]) == "@t":
    
        return inner_map(args[1])
    else:
        raise CarlaeEvaluationError("Mapping an element that is not a list")

def filter(args):
    func = args[0]

    def inner_filter(l):
        if l ==[]:
            return l
        elif l.tail == []:
            res = func([l.head])
            if res:
                return Pair(l.head,[])
            else:
                return []
        else:
            res = func([l.head])
            # print(res)
            if res:
                return Pair(l.head,inner_filter(l.tail))
            else:
                return inner_filter(l.tail)
    if is_linked_list([args[1]]) == "@t":
    
        return inner_filter(args[1])
    else:
        raise CarlaeEvaluationError("Filtering an element that is not a list")

def reduce(args):
    func = args[0]
    def inner_reduce(l,value):
        if l ==[]:
            return value
        elif l.tail == []:
            res = func([value,l.head])
            return res
        else:
            res = func([value,l.head])
            return inner_reduce(l.tail,res)
    list = args[1]
    initial = args[2]

    if is_linked_list([args[1]]) == "@t":
    
        return inner_reduce(list,initial)
    else:
        raise CarlaeEvaluationError("Filtering an element that is not a list")

def begin(args):
    return args[-1]

   
        

carlae_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": prod,
    "/": div,

    "@t": True,
    "@f": False,
    "=?": equal,
    ">": decreasing,
    ">=": non_increasing,
    "<": increasing,
    "<=": non_decreasing,
    "not": no,
    "pair": pair,
    "head": head,
    "tail": tail,
    "list":make_list,
    "list?": is_linked_list,
    "length": length_linked_list,
    "nth": get_index,
    "concat": concat,
    "map": map_func,
    "filter":filter,
    "reduce":reduce,
    "begin":begin,

}


##############
# Evaluation #
##############


def evaluate(tree, environment=None):
    """
    Evaluate the given syntax tree according to the rules of the Carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # print("tree", tree)
    if environment == None:
        environment = Environment(carlae_builtins)

    if type(tree) == int or type(tree) == float:
        return tree
    elif type(tree) == str:
        return environment.variable_lookup(tree)
    elif tree == []:
        raise CarlaeEvaluationError("The value is not in any environment")

    else:
        if tree[0] == ":=":
            variable = tree[1]

            if type(variable) == list:
                parameters = variable[1:]
                variable = variable[0]
                # print("name", variable)
                tree[1] = variable
                tree[2] = ['function', parameters, tree[2]]

            value = evaluate(tree[2], environment)

            environment.variables[variable] = value
            # print(environment.variables)
            return value

        elif tree[0] == "function":
            func = Function(tree[1], tree[2], environment)
            return func

        elif tree[0] == "and":
            if "@f" in tree[1:]:
                return False
            else:
                for i in tree[1:]:
                    res = evaluate(i,environment)
                    if res == False:
                        return False
                return True

        elif tree[0] == "or":
            if "@t" in tree[1:]:
                return True
            else:
                for i in tree[1:]:
                    res = evaluate(i,environment)
                    if res:
                        return True
                return False
        
        elif tree[0] == "if":
            res = evaluate(tree[1],environment)
            if res == True or res== "@t":
                return evaluate(tree[2],environment)
            else:
                return evaluate(tree[3],environment)
        
        elif tree[0] == "del":
            var = tree[1]
            if var in environment.variables:
                removed = environment.variables.pop(var)
                return removed
            else:
                raise CarlaeNameError("The variable is not defined,cannot delete")
        
        elif tree[0] == "let":
            rest = tree[1]
            new_env = Environment(environment)
            for variable in rest:
                var_name = variable[0]
                value = evaluate(variable[1],environment)
                new_env.variables[var_name] = value
            
            body = tree[2]
            return evaluate(body,new_env)
        
        elif tree[0] == "set!":
            var_name = tree[1]
            exp = evaluate(tree[2],environment)
            target_env = environment.variable_lookup_env(var_name)
            print(target_env.variables,var_name)
            
            
            target_env.variables[var_name] = exp
            
            return exp
            
        else:
            function = evaluate(tree[0], environment)
            if not callable(function):
                raise CarlaeEvaluationError
            args = []
            for i in tree[1:]:
                args.append(evaluate(i, environment))

            print("passed",args)
            
          
            return function(args)
def evaluate_file(loc,environment = None):

    if environment == None:
        environment = Environment(carlae_builtins)
    with open(loc,"r") as f:
        lines = f.readlines()
    s = ""
    for i in lines:
        s+=i
    value = evaluate(parse(tokenize(s)),environment)
    return value


class Pair():
    def __init__(self,head,tail) :
        self.head = head
        self.tail = tail
    
    def __str__(self):
        return "[" + str(self.head) + "," + str(self.tail) + "]"
        


class Environment():

    def __init__(self, parent):
        self.parent = parent
        self.variables = {'nil':[]}

    def variable_lookup(self, var):
        if var in self.variables:
            return self.variables[var]
        else:
            return self.variable_lookup_in_parent(var)

    def variable_lookup_in_parent(self, var):
        if type(self.parent) != dict:
            if var in self.parent.variables:
                return self.parent.variables[var]
            else:
                return self.parent.variable_lookup_in_parent(var)

        elif type(self.parent) == dict:
            if var in self.parent:
                return self.parent[var]
            else:
                raise CarlaeNameError("Not contained in the environment")
    

    def variable_lookup_env(self,var):
        
        if var in self.variables:
            return self
        else:
            return self.variable_lookup_parent_env(var)


    def variable_lookup_parent_env(self,var):
        
        if type(self.parent) != dict:
            print("parent variables",self.parent.variables)
            
            if var in self.parent.variables:
                
                return self.parent
            else:
                return self.parent.variable_lookup_parent_env(var)

        elif type(self.parent) == dict:
            
            if var in self.parent:
                return self.parent
            else:
                raise CarlaeNameError("Not contained in the environment or its parents")


def result_and_env(tree, environment=None):
    if environment == None:
        environment = Environment(carlae_builtins)

    value = evaluate(tree, environment)

    return (value, environment)


class Function():


    def __init__(self, parameters, body, environment):
        self.parameters = parameters
        self.body = body
        self.environment = environment
        print(body)
       

    def __call__(self, args):
        parameters = self.parameters
        body = self.body
        passed_parameters = args

        if len(parameters) != len(passed_parameters):
            raise CarlaeEvaluationError
        else:
            func_environment = Environment(self.environment)

            for i in range(len(args)):

                func_environment.variables[parameters[i]
                                           ] = passed_parameters[i]

            value = evaluate(body, func_environment)
            return value


def Repl():
    environment = Environment(carlae_builtins)
    args = sys.argv
    if len(args)>1:
        for i in args[1:]:
            evaluate_file(i,environment)
  
    while True:
        user_input = input(">")
        if user_input == "exit":
            break
        else:
            try:
                tokens = tokenize(user_input)
                parsed = parse(tokens)
                value, environment = result_and_env(parsed, environment)
                print(value)
            except CarlaeError as e:
                print(type(e).__name__, e)
                continue


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    Repl()
    # print(evaluate_file("test_files/small_test1.carlae"))
    
    # x = evaluate('nil')
    # y = evaluate('@f')
    # z=evaluate(0)
    # print(x == z)

# (:= (factorial n) (if (or (=? n 0) (=? n 1)) 1 (factorial (- n 1)) ) )