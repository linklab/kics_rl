import numpy as np
from ortools.linear_solver import pywraplp

from a_config import STATIC_TASK_RESOURCE_DEMAND_SAMPLE

# Create the solver
solver = pywraplp.Solver('simple_task_allocation', pywraplp.Solver.SAT_INTEGER_PROGRAMMING)


def solve_by_or_tool(num_tasks, num_resources, task_demands, resource_capacity):
    # Define the variables
    xs = []
    for i in range(num_tasks):
        xs.append(solver.IntVar(0, 1, 'x_' + str(i)))

    # Define the constraints
    for j in range(num_resources):
        solver.Add(solver.Sum([xs[i] * task_demands[i][j] for i in range(num_tasks)]) <= resource_capacity[j])

    # Define the objective function
    resources_of_selected_tasks = [xs[i] * task_demands[i][j] for j in range(num_resources) for i in range(num_tasks)]

    solver.Maximize(
        solver.Sum(resources_of_selected_tasks) / sum(resource_capacity)
    )

    # Solve the problem
    status = solver.Solve()

    # Print the solution
    utilization = None
    if status == pywraplp.Solver.OPTIMAL:
        utilization = solver.Objective().Value()
        selected_tasks_demand = np.zeros(shape=(num_resources,), dtype=float)
        for i in range(num_tasks):

            if xs[i].solution_value() == 1.0:
                for j in range(num_resources):
                    selected_tasks_demand[j] += task_demands[i][j]

            print("Task {0} [{1:>3},{2:>3}] is selected with value = {3} ({4}, {5})".format(
                i, task_demands[i][0], task_demands[i][1], xs[i].solution_value(),
                selected_tasks_demand[0], selected_tasks_demand[1]
            ))
        print("Utilization: ({0} + {1}) / ({2} + {3}) = {4}".format(
            selected_tasks_demand[0], selected_tasks_demand[1], resource_capacity[0], resource_capacity[1],
            sum(selected_tasks_demand) / sum(resource_capacity)
        ))
    else:
        print("Solver status: ", status)

    return utilization


if __name__ == "__main__":
    num_tasks = 10
    num_resources = 2

    use_random_task_demand = False

    if use_random_task_demand:
        task_demands = np.zeros(shape=(num_tasks, num_resources))

        for task_idx in range(num_tasks):
            task_demands[task_idx] = np.random.randint(
                low=[1] * num_resources, high=[100] * num_resources, size=(num_resources, )
            )
    else:
        task_demands = [
            [4, 3],
            [5, 6],
            [6, 8],
            [6, 9],
            [7, 11],
            [8, 15],
            [12, 16],
            [13, 16],
            [17, 17],
            [18, 18],
        ]

    resource_capacity = [50, 50]
    print("resource_capacity: ", resource_capacity)

    utilization = solve_by_or_tool(
        num_tasks=num_tasks, num_resources=2, task_demands=task_demands, resource_capacity=resource_capacity
    )

    print("Utilization: {0}".format(utilization))
