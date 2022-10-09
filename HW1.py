#---------FIRST HOMEWORK----------#
#submission by Simona De Risi 2081525


#Say "Hello, World!" With Python
import collections

if __name__ == '__main__':
    print("Hello, World!")


#Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    if not n % 2 == 0:
        print("Weird")
    else:
        if n <= 5:
            print("Not Weird")
        elif n <= 20:
            print("Weird")
        else:
            print("Not Weird")

#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print( a + b )
    print(a-b)
    print(a*b)


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(int(a/b))
    print(a/b)


#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)


#Write a function
def is_leap(year):
    leap = False

    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True

    return leap


#Print Function
if __name__ == '__main__':
    n = int(input())
    assert n>=1 and n<=150, 'Constraints not met'
    output = []
    for i in range(1,n+1):
        output.append(str(i))
    print(''.join(output))


#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    output = [[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i + j + k != n]
    print(output)


#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    list =[x for x in arr]
    massimo = max(list)
    list = [x for x in list if x != massimo]
    print(max(list))


#Nested Lists
if __name__ == '__main__':
    def ord(e):
        return e[0]

    records = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name, score])
    grades = []
    for c in records:
        if c[1] not in grades:
            grades.append(c[1])
    minimo = min(grades)
    grades.remove(minimo)
    records = [l for l in records if l[1] != minimo]
    minimo = min(grades)
    records.sort(key=ord)
    for l in records:
        if l[1] == minimo:
            print(l[0])


#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    query_scores = student_marks [query_name]
    average = 0
    for i in query_scores:
        average += i
    print("{:.2f}".format(average/3))


#Lists
if __name__ == '__main__':

    N = int(input())
    commands_list = []
    for i in range(N):
        commands_list.append(input().split())
    output_list = []
    for comm in commands_list:
        if comm[0] == 'insert':
            output_list.insert(int(comm[1]), int(comm[2]))
        elif comm[0] == 'print':
            print(output_list)
        elif comm[0] == 'remove':
            output_list.remove(int(comm[1]))
        elif comm[0] == 'append':
            output_list.append(int(comm[1]))
        elif comm[0] == 'sort':
            output_list.sort()
        elif comm[0] == 'pop':
            output_list.pop()
        elif comm[0] == 'reverse':
            output_list.reverse()

#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    tuples = tuple(integer_list)
    print(hash(tuples))


#sWAP cASE
def swap_case(s):
    return s.swapcase()

#String Split and Join
def split_and_join(line):
    output = line.split()
    return "-".join(output)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's your name?
def print_full_name(first, last):
    print('Hello', first, last + '! You just delved into python.')

#Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

#Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(0, len(string)):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count

#String Validators
if __name__ == '__main__':
    s = input()
    for i in range(0,len(s)):
        if s[i].isalnum():
            print("True")
            break
        elif i == len(s) - 1:
            print("False")
    for i in range(0,len(s)):
        if s[i].isalpha():
            print("True")
            break
        elif i == len(s) - 1:
            print("False")
    for i in range(0,len(s)):
        if s[i].isdigit():
            print("True")
            break
        elif i == len(s) - 1:
            print("False")
    for i in range(0,len(s)):
        if s[i].islower():
            print("True")
            break
        elif i == len(s) - 1:
            print("False")
    for i in range(0,len(s)):
        if s[i].isupper():
            print("True")
            break
        elif i == len(s) - 1:
            print("False")



#Text Wrap
def wrap(string, max_width):
    output = []
    for i in range(0, len(string), max_width):
        output.append(string[i:i+max_width])
        output.append('\n')
    return ''.join(output)

#Designer Door Mat
def print_row(vert, m):
    l = []
    for i in range(0,vert):
        l.append('.|.')
    return ''.join(l).center(m, "-")

sizes = input().split()
m = int(sizes[1])
i = (m-6)/3
rows = 0
while i > 1:
    rows += 1
    i -= 2
vert = 1
for i in range(0,rows+1):
    print(print_row(vert, m))
    vert += 2
print("WELCOME".center(m,"-"))
vert -= 2
for i in range(rows+1, 0, -1):
    print(print_row(vert, m))
    vert -= 2



#String Formatting
def print_line(n, w):
    line = [str(n).rjust(w, ' '),
            oct(n)[2:].rjust(w, ' '),
            hex(n)[2:].swapcase().rjust(w, ' '),
            bin(n)[2:].rjust(w, ' ')]
    print(' '.join(line))


def print_formatted(number):
    w = number.bit_length()
    for i in range(1, number + 1):
        print_line(i, w)


#Alphabet Rangoli
import string

def line(number, size):
    tot_lett = 2 * number - 1
    row = []
    l = size - 1
    for i in range(number):
        row.append(string.ascii_lowercase[l])
        row.append('-')
        l -= 1
    l += 2
    for i in range(1, number):
        row.append(string.ascii_lowercase[l])
        row.append('-')
        l += 1
    row.pop()
    return ''.join(row)


def print_rangoli(size):
    width = 4 * size - 3
    rows = 2 * size - 1
    out = []
    for i in range(size):
        out.append(line(i + 1, size).center(width, '-'))
    for i in range(size):
        print(out[i])
    for i in range(size - 2, -1, -1):
        print(out[i])


#Capitalize!
def solve(s):
    l = s.split()
    out = [n[0].upper() + n[1:] for n in l]
    output = ' '.join(out)
    return output

#The Minion Game
#doesn't work because of time limit exceeded
def count_occurrences(sub, string):
    count = 0
    l = len(sub)
    for i in range(len(string) - l + 1):
        if string[i:i + l] == sub:
            count += 1
    return count

def minion_game(string):
    vowels = 'AEIOU'
    stuart = 0
    kevin = 0
    s = []
    k = []
    for i in range(1, len(string) + 1): #numero di lettere considerato
        for j in range(len(string) - i + 1): #iterazione sulla stringa
            sub = string[j:j + i]
            if sub[0] in vowels:
                if not sub in k:
                    k.append(sub)
                    kevin += count_occurrences(sub, string)
            else:
                if not sub in s:
                    s.append(sub)
                    stuart += count_occurrences(sub, string)
    if kevin == stuart:
        print('Draw')
    elif kevin > stuart:
        print('Kevin', kevin)
    else:
        print('Stuart', stuart)


#Merge the Tools
def merge_the_tools(string, k):
    for i in range(0, int(len(string)/k)):
        s = string[i*k:(i*k)+k]
        output = []
        for j in s:
            if not j in output:
                output.extend(j)
        print(''.join(output))


#Text Alignment
n = int(input())

liv = int(n / 2)
spaces = 3 * n
width = 5 * n

for i in range(1, n + 1):
    print(('H' * (2 * i - 1)).center(2 * n - 1, ' '))

for i in range(1, n + 2):
    print(((' ' * spaces).center(width, 'H')).rjust(width + liv, ' '))

for i in range(int((n + 2) / 2)):
    print(('H' * width).rjust(width + liv, ' '))

for i in range(1, n + 2):
    print(((' ' * spaces).center(width, 'H')).rjust(width + liv, ' '))

for i in range(n, 0, -1):
    print((('H' * (2 * i - 1)).center(2 * n - 1, ' ')).rjust(width + (2 * liv), ' '))


#Introduction to Sets
def average(array):
    it = set(arr)
    sums = sum(it)
    return sums/len(it)

#No Idea!
n_and_m = list(map(int, input().split()))
n = n_and_m[0]
m = n_and_m[1]
arr = list(map(int, input().split()))
a = set((map(int, input().split())))
b = set((map(int, input().split())))

happiness = 0
for i in arr:
    if i in a:
        happiness += 1
    if i in b:
        happiness -= 1
print(happiness)


#Symmetric Difference
m = int(input())
arr_m = set(list(map(int, input().split())))
n = int(input())
arr_n = set(list(map(int, input().split())))
output = set(arr_m)
for i in arr_n:
    if i in output:
        output.discard(i)
    else:
        output.add(i)
for i in sorted(output):
    print(i)



#Set .add()
n = int(input())
s = set()
for i in range(0, n):
    s.add(input())
print(len(s))


#Set . union() Operation
n = int(input())
n_stud = set(list(map(int, input().split())))
b = int(input())
b_stud = set(list(map(int, input().split())))
out = n_stud.union(b_stud)
print(len(out))

#Set .intersection() Operation
n = int(input())
n_stud = set(list(map(int, input().split())))
b = int(input())
b_stud = set(list(map(int, input().split())))
out = n_stud.intersection(b_stud)
print(len(out))


#Set .difference() Operation
n = int(input())
n_stud = set(list(map(int, input().split())))
b = int(input())
b_stud = set(list(map(int, input().split())))
out = n_stud.difference(b_stud)
print(len(out))


#Set .symmetric_difference() Operation
n = int(input())
n_stud = set(list(map(int, input().split())))
b = int(input())
b_stud = set(list(map(int, input().split())))
out = n_stud.symmetric_difference(b_stud)
print(len(out))

#Set Mutations
a = int(input())
set_a = set(list(map(int, input().split())))
n = int(input())
for i in range(0, n):
    comm = input().split()
    set_b = set(list(map(int, input().split())))
    if comm[0] == 'update':
        set_a.update(set_b)
    elif comm[0] == 'intersection_update':
        set_a.intersection_update(set_b)
    elif comm[0] == 'difference_update':
        set_a.difference_update(set_b)
    elif comm[0] == 'symmetric_difference_update':
        set_a.symmetric_difference_update(set_b)
out = 0
for i in set_a:
    out += i
print(out)


#The Captain's Room
from collections import Counter

n = int(input())
rooms = list(map(int, input().split()))
c = Counter(rooms)
r = c.most_common()[-1]
print(r[0])

#Check Subset
t = int(input())

for i in range(0, t):
    n_a = int(input())
    set_a = set(list(map(int, input(). split())))
    n_b = int(input())
    set_b = set(list(map(int, input(). split())))
    if len(set_a.intersection(set_b)) == n_a:
        print('True')
    else:
        print('False')

#Check Strict Superset
set_a = set(list(map(int, input().split())))
n = int(input())
out = True
for i in range(0,n):
    set_b = set(list(map(int, input().split())))
    if not (out and len(set_a.intersection(set_b)) == len(set_b) and len(set_a.difference(set_b)) > 0):
        out = False
        break
print(out)

#Set .discard() .remove() & .pop()
n = int(input())
my_set = set(map(int, input().split()))
n = int(input())
for i in range(n):
    comm = input().split()
    if comm[0] == 'pop':
        my_set.pop()
    elif comm[0] == 'discard':
        my_set.discard(int(comm[1]))
    elif comm[0] == 'remove':
        my_set.remove(int(comm[1]))
print(sum(my_set))


#collection.Counter()
from collections import Counter

x = int(input())
sizes = list(map(int, input().split()))
counter = Counter(sizes)
n = int(input())
earning = 0
for i in range(n):
    request = input().split()
    size = int(request[0])
    price = int(request[1])
    if counter[size] > 0 :
        earning += price
        counter[size] -= 1
print(earning)


#DefaultDict Tutorial
from collections import defaultdict

n_and_m = list(map(int, input().split()))
n = n_and_m[0]
m = n_and_m[1]
d = defaultdict(list)
for i in range(n):
    d[input()].append(i+1)
keys = d.keys()
for j in range(m):
    w = input()
    if w in keys:
        sd = [str(a) for a in d[w]]
        print(' '.join(sd))
    else:
        print('-1')


#Collections.namedtuple()
from collections import namedtuple

n = int(input())
keys = input().split()
for i in range(len(keys)):
    if keys[i] == 'ID':
        t = i
    if keys[i] == 'MARKS':
        u = i
    if keys[i] == 'NAME':
        v = i
    if keys[i] == 'CLASS':
        z = i
Spreadsheet = namedtuple('Spreadsheet', ' '.join(keys))
students = []
for i in range(n):
    values = input().split()
    a = Spreadsheet(ID=values[t], MARKS=int(values[u]), NAME=values[v], CLASS=values[z])
    students.append(a)
sums = 0
for i in range(n):
    sums += students[i].MARKS
print(format(sums / n, '.2f'))

#Collections.OrderedDict()
from collections import OrderedDict

ordered_dict = OrderedDict()
n = int(input())
for i in range(n):
    data = input().split()
    l = len(data)
    name = ' '.join(data[0:l-1])
    price = int(data[l-1])
    if name not in ordered_dict:
        ordered_dict[name] = price
    else:
        ordered_dict[name] += price
for item in ordered_dict.keys():
    print(item, ordered_dict[item])

#Word Order
from collections import OrderedDict

n = int(input())
l = OrderedDict()
for i in range(n):
    w = input()
    if w in l:
        l[w] += 1
    else:
        l[w] = 1
print(len(l))
for w in l:
    print(l[w], end=' ')

#Collections.deque()
from collections import deque

n = int(input())
d = deque()
for i in range(n):
    comm = input().split()
    if comm[0] == 'append':
        d.append(int(comm[1]))
    elif comm[0] == 'pop':
        d.pop()
    elif comm[0] == 'popleft':
        d.popleft()
    elif comm[0] == 'appendleft':
        d.appendleft(int(comm[1]))
for el in d:
    print(el, end=' ')

#Piling Up!
from collections import deque

def is_stackable(n, blocks):
    if blocks[-1] >= blocks[0]:
        act = blocks[-1]
        blocks.pop()
    else:
        act = blocks[0]
        blocks.popleft()
    for i in range(n-1):
        if act >= blocks[0] >= blocks[-1]:
            act = blocks[0]
            blocks.popleft()
        elif act >= blocks[-1] >= blocks[0]:
            act = blocks[-1]
            blocks.pop()
        elif act >= blocks[0] and act < blocks[-1]:
            act = blocks[0]
            blocks.popleft()
        elif act >= blocks[-1] and act < blocks[0]:
            act = blocks[-1]
            blocks.pop()
        elif act < blocks[0] and act < blocks[-1]:
            break
    if len(blocks) == 0:
        return True
    else:
        return False

t = int(input())
output = []
for i in range(t):
    n = int(input())
    blocks = deque()
    blocks.extend(map(int, input().split()))
    if is_stackable(n, blocks):
        output.append('Yes')
    else:
        output.append('No')
for el in output:
    print(el)


#Company Logo
from collections import Counter

if __name__ == '__main__':
    s = input()
    counter = Counter(s)
    mc = counter.most_common()
    out = []
    i = 0
    while len(out) < 3:
        l = [c for c in mc if c[1] == mc[i][1]]
        l.sort(key = lambda x: x[0])
        out.extend(l)
        i += len(l)
    for i in range(3):
        print(out[i][0], out[i][1])


#Time Delta


#Calendar Module
import calendar

days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
date = list(map(int, input().split()))
day = calendar.weekday(date[2], date[0], date[1])
print(days[day])

#Exceptions
n = int(input())

for i in range(n):
    try:
        inp = list(map(int, input().split()))
    except ValueError as e:
        print('Error Code:', e)
        continue
    try:
        print(inp[0]//inp[1])
    except ZeroDivisionError:
        print('Error Code: integer division or modulo by zero')

#Zipped!
nx = list(map(int, input().split()))
marks = []
for i in range(nx[1]):
    marks.append(list(map(float, input().split())))
zipped = list(zip(*marks))
for i in range(nx[0]):
    somma = sum(zipped[i])
    print(format(somma/nx[1], '.1f'))

#Athlete Sort
if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr.sort(key=lambda x: x[k])
    for t in arr:
        print(' '.join(list(map(str, t))))

#ginortS
s = input()
lc = []
uc = []
odd = []
even =[]
for c in s:
    if c.islower():
        lc.append(c)
    elif c.isupper():
        uc.append(c)
    elif c.isdigit():
        if int(c) % 2 ==0:
            even.append(c)
        else:
            odd.append(c)
lc.sort()
uc.sort()
odd.sort()
even.sort()
print(''.join(lc+uc+odd+even))


#Map and Lambda Function
cube = lambda x: x**3
fib = lambda x: x[-1] + x[-2]

def fibonacci(n):
    if n == 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        out = [0, 1]
        for i in range(3,n+1):
            out.append(fib(out))
        return out


#Detect Floating Point Number
import re

t = int(input())
out = []
pattern = r'([+-]\d+)?\.\d+'
for i in range(t):
    s = input()
    out.append(bool(re.fullmatch(pattern, s)))
for a in out:
    print(a)

#re.split()
regex_pattern = r"[,\.]"


#Group(), Groups() & Groupdict()


#Re.findall() & Re.finditer()


#Re.start() & Re.end()


#Regex Substitution


#Valoidating Roman Numerals


#Validating phone numbers


#Validating and Parsing Email Addresses


#Hex Color Code


#HTML Parser - Part 1


#HTML Parser - Part 1


#Detect HTML Tags, Attributes and Attribute Values


#Validating UID


#Validating Credit Card Numbers


#Validating Postal Codes


#Matrix Script



#XML 1 - Find the Score
def get_attr_number(node):
    if len(node.tag) == 0:
        return len(node.attrib)
    else:
        ch = 0
        for n in node:
            ch += get_attr_number(n)
        return ch + len(node.attrib)

#XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    if len(elem) == 0:
        if (level + 1) > maxdepth:
            maxdepth = level + 1
        return
    else:
        for n in elem:
            depth(n, level+1)

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        out = []
        for n in l:
            out.append('+91' + ' ' + n[-10:-5] + ' ' + n[-5:])
        out.sort()
        for n in out:
            print(n)
    return fun

#Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        people.sort(key = lambda x: int(x[2]))
        out = []
        for p in people:
            out.append(f(p))
        return out
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

#Arrays
def arrays(arr):
    a = numpy.array(arr, float)
    return (a[::-1])


#Shape and Reshape
import numpy

arr = list(map(int, input().split()))
print (numpy.reshape(arr, (3,3)))


#Transpose and Flatten
import numpy

nm = list(map(int, input().split()))
n = nm[0]
m = nm[1]
inp = []
for i in range(n):
    inp.append(list(map(int, input().split())))
arr = numpy.array(inp)
print(numpy.transpose(arr))
print(arr.flatten())


#Concatenate
import numpy

nmp = list(map(int, input(). split()))
n = nmp[0]
m = nmp[1]
arr_n = []
arr_m = []
for i in range(n):
    arr_n.append(list(map(int, input(). split())))
for i in range(m):
    arr_m.append(list(map(int, input(). split())))
narr_n = numpy.array(arr_n)
narr_m = numpy.array(arr_m)
print(numpy.concatenate((narr_n, narr_m),axis=0))

#Zeros and Ones
import numpy

shape = tuple(list(map(int, input().split())))
zero_list = numpy.zeros(shape, dtype = numpy.int)
one_list = numpy.ones(shape, dtype = numpy.int)
print(zero_list)
print(one_list)

#Eye and Identity
import numpy
numpy.set_printoptions(legacy = '1.13')

nm = list(map(int, input().split()))
n = nm[0]
m = nm[1]
print(numpy.eye(n, m))

#Array Mathematics
import numpy

nm = list(map(int, input().split()))
n = nm[0]
m = nm[1]
a = []
b = []
for i in range(n):
    a.append(list(map(int, input().split())))
for i in range(n):
    b.append(list(map(int, input().split())))
arr_a = numpy.array(a)
arr_b = numpy.array(b)
print(arr_a + arr_b)
print(arr_a - arr_b)
print(arr_a * arr_b)
print(numpy.floor_divide(a, b))
print(arr_a % arr_b)
print(arr_a ** arr_b)

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy = '1.13')

l = list(map(float, input().split()))
arr = numpy.array(l)
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

#Sum and Prod
import numpy

nm = list(map(int, input().split()))
n = nm[0]
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
sum_arr = numpy.sum(arr, axis = 0)
print(numpy.prod(sum_arr))



#Min and Max
import numpy

nm = list(map(int, input().split()))
n = nm[0]
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
mini = numpy.min(arr, axis = 1)
print(numpy.max(mini))

#Mean, Var and Std
import numpy

nm = list(map(int, input().split()))
n = nm[0]
m = nm[1]
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(round(numpy.std(arr), 11))

#Dot and Cross
import numpy

n = int(input())
l1 = []
l2 = []
for i in range(n):
    l1.append(list(map(int, input().split())))
for i in range(n):
    l2.append(list(map(int, input().split())))
a = numpy.array(l1)
b = numpy.array(l2)
b = b.T
out = []
for i in range(n):
    out.append( [ numpy.dot(a[i],b[j]) for j in range(n) ] )
print(numpy.array(out))



#Inner and Outer
import numpy

l1 = list(map(int, input().split()))
l2 = list(map(int, input().split()))
a = numpy.array(l1)
b = numpy.array(l2)
print(numpy.inner(a, b))
print(numpy.outer(a, b))

#Polynomials
import numpy

coeff = list(map(float, input().split()))
x = float(input())
print(numpy.polyval(coeff, x))

#Linear Algebra
import numpy

n = int(input())
l = []
for i in range(n):
    l.append(list(map(float, input().split())))
print(round(numpy.linalg.det(numpy.array(l)), 2))



#Birthday Cake Candles
from collections import Counter

def birthdayCakeCandles(candles):
    c = Counter(candles)
    return c[max(candles)]


#Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if (x2 > x1 and v2 > v1) or (x1 > x2 and v1 > v2) or (not x1 == x2 and v1 == v2):
        return 'NO'
    dist = abs(x2 - x1)
    while not x2 == x1:
        x2 += v2
        x1 += v1
        new_dist = abs(x2 - x1)
        if new_dist > dist:
            return 'NO'
        dist = new_dist
    return 'YES'


#Viral Advertising
def viralAdvertising(n):
    i = 2
    shared = 5
    liked = 2
    cumulative = 2
    while i <=n :
        shared = 3 * (shared//2)
        liked = (shared//2)
        cumulative += liked
        i += 1
    return cumulative

#Recursive Digit Sum
def superDigit(n, k):
    if (len(n)==1 and k ==1):
        return int(n)
    numb = [int(v) for v in n]
    print(numb)
    return superDigit(str(sum(numb)*k), 1)


#Insertion Sort - Part 1
def insertionSort1(n, arr):
    to_be_positioned = arr[n-1]
    i = n-2
    while(i>=0):
        if arr[i] <= to_be_positioned:
            arr[i+1] = to_be_positioned
            print(' '.join(map(str, arr)))
            break
        else:
            arr[i+1] = arr[i]
            i -= 1
            print(' '.join(map(str, arr)))
    if i == -1:
        arr[0] = to_be_positioned
        print(' '.join(map(str, arr)))


#Insertion Sort - Part 2
def insertionSort2(n, arr):
    for i in range(1, n):
        if arr[i-1] <= arr[i]:
            print(' '.join(map(str, arr)))
        else:
            j = i - 1
            while(arr[i] < arr[j-1] and j>0):
                j -= 1
            act = arr[i]
            for z in range(i, j, -1):
                arr[z] = arr[z-1]
            arr[j] = act
            print(' '.join(map(str, arr)))