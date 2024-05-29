import numpy as np
from ortools.linear_solver import pywraplp

from a_config import STATIC_TASK_RESOURCE_DEMAND_SAMPLE

# Create the solver
solver = pywraplp.Solver('simple_task_allocation', pywraplp.Solver.SAT_INTEGER_PROGRAMMING)


def solve(num_tasks, num_resources, task_demands, task_values, resource_capacity):
    # Define the variables
    xs = []
    for n in range(num_tasks):
        xs.append(solver.IntVar(0, 1, 'x_' + str(n)))

    # Define the constraints
    for m in range(num_resources):
        solver.Add(solver.Sum([xs[n] * task_demands[n][m] for n in range(num_tasks)]) <= resource_capacity[m])

    # Define the objective function
    total_value = [xs[n] * task_values[n] for n in range(num_tasks)]
    solver.Maximize(solver.Sum(total_value))

    # Solve the problem
    status = solver.Solve()

    # Print the solution
    if status == pywraplp.Solver.OPTIMAL:
        total_value = solver.Objective().Value()
        selected_task_value = np.zeros(shape=(num_tasks,), dtype=int)
        selected_task_demand = np.zeros(shape=(num_resources,), dtype=int)
        for i in range(num_tasks):

            if xs[i].solution_value() == 1.0:
                selected_task_value[i] = task_values[i]
                for j in range(num_resources):
                    selected_task_demand[j] += task_demands[i][j]

            print("Task {0} [{1:>3},{2:>3}] / [{3:>3}] is {4:<12} ([{5:>3},{6:>3}] / [{7:>3}])".format(
                i, task_demands[i][0], task_demands[i][1], task_values[i],
                "selected" if xs[i].solution_value() == 1.0 else "not selected",
                selected_task_demand[0], selected_task_demand[1],
                sum(selected_task_value)
            ))
    else:
        total_value = None
        print("Solver status: ", status)

    return total_value


if __name__ == "__main__":
    num_tasks = 10
    num_resources = 2

    use_random_task_demand = True

    if use_random_task_demand:
        task_demands = np.zeros(shape=(num_tasks, num_resources), dtype=int)
        task_values = np.zeros(shape=(num_tasks,), dtype=int)
        for task_idx in range(num_tasks):
            task_demands[task_idx] = np.random.randint(
                low=[50] * num_resources, high=[100] * num_resources, size=(num_resources,)
            )
            task_values[task_idx] = np.random.randint(
                low=[1] * num_resources, high=[100] * num_resources, size=(1,)
            )
    else:
        task_demands = [
            [54, 53],
            [65, 96],
            [56, 78],
            [65, 92],
            [71, 51],
            [68, 65],
            [52, 86],
            [83, 86],
            [77, 87],
            [58, 98],
        ]
        task_values = [
            12, 22, 39, 98, 55, 19, 23, 76, 44, 81
        ]

    resource_capacity = [300, 300]
    print("resource_capacity: ", resource_capacity)

    total_value = solve(
        num_tasks=num_tasks, num_resources=2, task_demands=task_demands,
        task_values=task_values,
        resource_capacity=resource_capacity
    )

    print("Total Value: {0}".format(total_value))
