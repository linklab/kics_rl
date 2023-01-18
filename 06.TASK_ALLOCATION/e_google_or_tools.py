import numpy as np
from ortools.linear_solver import pywraplp

# Create the solver
solver = pywraplp.Solver('multi_dimensional_knapsack', pywraplp.Solver.SAT_INTEGER_PROGRAMMING)


def solve_by_or_tool(num_tasks, num_resources, task_demands, capacity_of_resources):
    # Define the variables
    # num_tasks = 5
    # num_resources = 3
    items = []
    for i in range(num_tasks):
        items.append(solver.IntVar(0, 1, 'item_' + str(i)))

    # Define the constraints
    # task_demands = [[4, 2, 3], [3, 5, 2], [2, 1, 3], [4, 2, 1], [2, 3, 3]]
    # capacity_of_resources = [10, 15, 12]
    for j in range(num_resources):
        solver.Add(solver.Sum([items[i] * task_demands[i][j] for i in range(num_tasks)]) <= capacity_of_resources[j])

    # Define the objective function
    resources_of_selected_tasks = [items[i] * task_demands[i][j] for j in range(num_resources) for i in range(num_tasks)]
    solver.Maximize(
        solver.Sum(resources_of_selected_tasks) / sum(capacity_of_resources)
    )

    # Solve the problem
    status = solver.Solve()

    # Print the solution
    utilization = None
    if status == pywraplp.Solver.OPTIMAL:
        utilization = solver.Objective().Value()
        # print("Optimal solution found with objective value = ", solver.Objective().Value())
        # for i in range(num_tasks):
        #     if items[i].solution_value() > 0:
        #         print("Item", i, "is selected with value = ", values[i])
    else:
        print("Solver status: ", status)

    return utilization


if __name__ == "__main__":
    num_tasks = 100000
    num_resources = 2

    task_demands = np.zeros(shape=(num_tasks, num_resources))

    for task_idx in range(num_tasks):
        task_demands[task_idx] = np.random.randint(
            low=[1] * num_resources,
            high=[20] * num_resources,
            size=(1, num_resources)
        )
    capacity_of_resources = [100] * num_resources

    or_tool_solution = solve_by_or_tool(
        num_tasks=num_tasks,
        num_resources=2,
        task_demands=task_demands,
        capacity_of_resources=capacity_of_resources
    )

    print(or_tool_solution)