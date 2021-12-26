import itertools
import numpy as np
from numpy import random
from scipy.optimize import linear_sum_assignment


# Призначення задачі
class TaskAssignment:

    # Ініціалізація класу, обов'язковими вхідними параметрами є матриця задач та метод розподілу, серед яких є два методи розподілу, метод permutation або метод Угорщини.
    def __init__(self, task_matrix, func):
        self.task_matrix = task_matrix
        self.func = func
        if func == 'permutation':
            self.min_cost, self.top_answ = self.permutation(task_matrix)
        if func == 'Hungary':
            self.min_cost, self.top_answ = self.Hungary(task_matrix)

        # Повний метод аранжування

    def permutation(self, matrix):
        number_of_choice = len(matrix)
        answ = []
        values = []
        for solution_ in itertools.permutations(range(number_of_choice)):
            solution_ = list(solution_)
            solution = []
            value = 0
            for i in range(len(matrix)):
                value += matrix[i][solution_[i]]
                solution.append(task_matrix[i][solution_[i]])
            values.append(value)
            answ.append(solution)
        min_cost = np.min(values)
        top_answ = answ[values.index(min_cost)]
        return min_cost, top_answ

        # Угорський метод

    def Hungary(self, task_matrix):
        b = task_matrix.copy()
        # Рядок та стовпець мінус 0 0
        for i in range(len(b)):
            row_min = np.min(b[i])
            for j in range(len(b[i])):
                b[i][j] -= row_min
        for i in range(len(b[0])):
            col_min = np.min(b[:, i])
            for j in range(len(b)):
                b[j][i] -= col_min
        line_count = 0
        # Коли кількість рядків менше довжини матриці, цикл
        while (line_count < len(b)):
            line_count = 0
            row_zero_count = []
            col_zero_count = []
            for i in range(len(b)):
                row_zero_count.append(np.sum(b[i] == 0))
            for i in range(len(b[0])):
                col_zero_count.append((np.sum(b[:, i] == 0)))
                # Вибрати порядок (гілка або стовпець)
            line_order = []
            row_or_col = []
            for i in range(len(b[0]), 0, -1):
                while (i in row_zero_count):
                    line_order.append(row_zero_count.index(i))
                    row_or_col.append(0)
                    row_zero_count[row_zero_count.index(i)] = 0
                while (i in col_zero_count):
                    line_order.append(col_zero_count.index(i))
                    row_or_col.append(1)
                    col_zero_count[col_zero_count.index(i)] = 0
                    # Намалюйте лінію, що покриває 0, і отримайте матрицю після рядка мінус мінімальне значення та стовпець плюс мінімальне значення
            delete_count_of_row = []
            delete_count_of_rol = []
            row_and_col = [i for i in range(len(b))]
            for i in range(len(line_order)):
                if row_or_col[i] == 0:
                    delete_count_of_row.append(line_order[i])
                else:
                    delete_count_of_rol.append(line_order[i])
                c = np.delete(b, delete_count_of_row, axis=0)
                c = np.delete(c, delete_count_of_rol, axis=1)
                line_count = len(delete_count_of_row) + len(delete_count_of_rol)
                # Коли кількість рядків дорівнює довжині матриці, вискакуємо
                if line_count == len(b):
                    break
                    # Визначаємо, чи потрібно малювати лінію, щоб покрити всі нулі, якщо вона покриває, операції складання та віднімання
                if 0 not in c:
                    row_sub = list(set(row_and_col) - set(delete_count_of_row))
                    min_value = np.min(c)
                    for i in row_sub:
                        b[i] = b[i] - min_value
                    for i in delete_count_of_rol:
                        b[:, i] = b[:, i] + min_value
                    break
        row_ind, col_ind = linear_sum_assignment(b)
        min_cost = task_matrix[row_ind, col_ind].sum()
        top_answ = list(task_matrix[row_ind, col_ind])
        return min_cost, top_answ


def min_(zero_mat, mark_zero):

    min_row = [99999, -1]

    for row in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row] == True) > 0 and min_row[0] > np.sum(zero_mat[row] == True):
            min_row = [np.sum(zero_mat[row] == True), row]


    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False


def _matrix(mat):

    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()


    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_(zero_bool_mat_copy, marked_zero)


    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])


    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):

                if row_array[j] == True and j not in marked_cols:

                    marked_cols.append(j)
                    check_switch = True

        for row, coll in marked_zero:

            if row not in non_marked_row and coll in marked_cols:

                non_marked_row.append(row)
                check_switch = True

    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

    return (marked_zero, marked_rows, marked_cols)


def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []


    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    min_num = min(non_zero_element)


    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num

    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
    return cur_mat


def hungarian_(mat):
    dim = mat.shape[0]
    cur_mat = mat


    for row in range(mat.shape[0]):
        cur_mat[row] = cur_mat[row] - np.min(cur_mat[row])

    for coll in range(mat.shape[1]):
        cur_mat[:, coll] = cur_mat[:, coll] - np.min(cur_mat[:, coll])
    zero_count = 0
    while zero_count < dim:

        ans, marked_rows, marked_cols = _matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

    return ans


def answer(mat, position):

    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(position)):
        total += mat[position[i][0], position[i][1]]
        ans_mat[position[i][0], position[i][1]] = mat[position[i][0], position[i][1]]
    return total, ans_mat


def main():

    cost = np.array([[7, 6, 2, 9, 2],
                            [4, 2, 1, 3, 9],
                            [5, 3, 8, 9, 5],
                            [6, 8, 2, 8, 6],
                            [9, 5, 6, 1, 7]])
    ans = hungarian_(cost.copy())
    ans, ans_mat = answer(cost, ans)

    # Show the result
    print(f"Результат задачі 1 = {ans:.0f}\n{ans_mat}")


    profit = np.array([[7, 6, 2, 9, 2],
                              [5, 2, 1, 3, 9],
                              [5, 7, 8, 9, 5],
                              [6, 8, 8, 8, 6],
                              [9, 5, 6, 10, 1]])
    max_value = np.max(profit)
    cost = max_value - profit
    ans = hungarian_(cost.copy())
    ans, ans_mat = answer(profit, ans)

    print(f"Результат задачі 2 = {ans:.0f}\n{ans_mat}")


if __name__ == '__main__':

    main()
    rd = random.RandomState(10000)
    task_matrix = rd.randint(0, 100, size=(5, 5))
    # Используйте метод полного размещения, чтобы добиться распределения задач
    ass_by_per = TaskAssignment(task_matrix, 'permutation')

    # Використовуйте Угорський метод для розподілу завдань
    ass_by_Hun = TaskAssignment(task_matrix, 'Hungary')
    print('cost matrix = ', '\n', task_matrix)
    print('Назначение задачи для всех способов размещения:')
    print('min cost = ', ass_by_per.min_cost)
    print('best solution = ', ass_by_per.top_answ)
    print('Назначение задачи по венгерскому методу:')
    print('min cost = ', ass_by_Hun.min_cost)
    print('best solution = ', ass_by_Hun.top_answ)