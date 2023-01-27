import numpy as np
from ortools.linear_solver import pywraplp

from a_config import STATIC_TASK_RESOURCE_DEMAND_SAMPLE

# Create the solver
solver = pywraplp.Solver('simple_task_allocation', pywraplp.Solver.SAT_INTEGER_PROGRAMMING)


def solve_by_or_tool(num_tasks, num_resources, task_demands, resource_capacity):
    # Define the variables
    items = []
    for i in range(num_tasks):
        items.append(solver.IntVar(0, 1, 'item_' + str(i)))

    # Define the constraints
    for j in range(num_resources):
        solver.Add(solver.Sum([items[i] * task_demands[i][j] for i in range(num_tasks)]) <= resource_capacity[j])

    # Define the objective function
    resources_of_selected_tasks = [items[i] * task_demands[i][j] for j in range(num_resources) for i in range(num_tasks)]

    solver.Maximize(
        solver.Sum(resources_of_selected_tasks) / sum(resource_capacity)
    )

    # Solve the problem
    status = solver.Solve()

    # Print the solution
    utilization = None
    if status == pywraplp.Solver.OPTIMAL:
        utilization = solver.Objective().Value()
        for i in range(num_tasks):
            print("Task {0} [{1:>3},{2:>3}] is selected with value = {3}".format(
                i, task_demands[i][0], task_demands[i][1], items[i].solution_value())
            )
        resources_of_selected_tasks = [
            items[i].solution_value() * task_demands[i][j] for j in range(num_resources) for i in range(num_tasks)
        ]
        print(sum(resources_of_selected_tasks) / sum(resource_capacity))
    else:
        print("Solver status: ", status)

    return utilization


if __name__ == "__main__":
    num_tasks = 10
    num_resources = 2

    use_random_task_demand = True

    if use_random_task_demand:
        task_demands = np.zeros(shape=(num_tasks, num_resources))

        for task_idx in range(num_tasks):
            task_demands[task_idx] = np.random.randint(
                low=[1] * num_resources, high=[100] * num_resources, size=(num_resources, )
            )
    else:
        task_demands = STATIC_TASK_RESOURCE_DEMAND_SAMPLE

    resource_capacity = [num_tasks * 50] * num_resources
    print("resource_capacity: ", resource_capacity)

    utilization = solve_by_or_tool(
        num_tasks=num_tasks, num_resources=2, task_demands=task_demands, resource_capacity=resource_capacity
    )

    print("Utilization: {0}".format(utilization))
