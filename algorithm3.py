import numpy as np
import random

# ============================== inputs ==============================

cache_size = 2
number_of_items = 10
number_of_users = 5

e_i_u = np.random.randint(0, 4, (number_of_items, number_of_users))
# print('e_i_u', e_i_u)
e_i_u_v = np.random.randint(0, 4, (number_of_items, number_of_users, number_of_users))
# print('e_i_u_v', e_i_u_v)


def solve_subproblem_a(x, y):
    cached_items = np.count_nonzero(x)
    if cached_items > 1:
        y_i = np.random.randint(0, 2, (cached_items, number_of_users), dtype='bool')
    else:
        y_i = np.random.randint(0, 2, number_of_users, dtype='bool')
    return y_i


def objective_function(item, y):
    return random.randint(0,20)


# ============================== step 0 ==============================
k = 0

x_0 = np.full(number_of_items, False)
x_0[0:cache_size] = 1
random.shuffle(x_0)
# print('x_0', x_0)

# ============================== step 1 ==============================
y_0_i = solve_subproblem_a(x_0, np.full(number_of_users, True))
# print('y_0_i', y_0_i)

# ============================== step 2 ==============================
y_0 = y_0_i.any(axis=0)
# print('y_0', y_0)

temp_y = []
a_0_i = []
for i in range(0, number_of_items):
    temp_x = np.full(number_of_items, False)
    temp_x[i] = True

    temp_y_i = solve_subproblem_a(temp_x, y_0)
    # print(temp_y_i)
    temp_y.append(temp_y_i)
    # print('temp_y', temp_y)
    a_0_i.append(objective_function(i, temp_y_i))

temp_y = np.asarray(temp_y)
# print('temp_y', temp_y)
print('a_0_i', a_0_i)

a_i_previous = a_0_i
x_previous = x_0
y_i_previous = y_0_i

while True:
    print('Step No', k)

    # ============================== step 3 ==============================
    k += 1

    # ============================== step 4 ==============================
    x_current = np.full(number_of_items, False)
    for i in range(0, cache_size):
        next_a_max_index = np.where(a_i_previous == np.amax(a_i_previous))
        # print(next_a_max_index)
        next_a_max_index = next_a_max_index[0][0]
        # print(next_a_max_index)
        x_current[next_a_max_index] = True
        a_i_previous[next_a_max_index] = -1
    # print('x_current', x_current)

    # ============================== step 5 ==============================
    y_i_current = solve_subproblem_a(x_current, np.full(number_of_users, True))

    # print and stop while
    print('y_i_current', y_i_current)
    print('x_current', x_current)

    if np.array_equal(y_i_current, y_i_previous) and np.array_equal(x_current, x_previous):
        break

    y_current = y_i_current.any(axis=0)
    # print('y_current', y_current)

    temp_y = []
    a_i_current = []
    for i in range(0, number_of_items):
        temp_x = np.full(number_of_items, False)
        temp_x[i] = True

        temp_y_i = solve_subproblem_a(temp_x, y_current)
        # print(temp_y_i)
        temp_y.append(temp_y_i)
        # print('temp_y', temp_y)
        a_i_current.append(objective_function(i, temp_y_i))

    temp_y = np.asarray(temp_y)
    # print('temp_y', temp_y)
    # print('a_1_i', a_1_i)

    a_i_previous = a_i_current
    x_previous = x_current
    y_i_previous = y_current
