# Import necessary modules
from pydrake.all import (EqualityConstrainedQPSolver, 
                         MathematicalProgram, Solve)

# Create a mathematical program
prog = MathematicalProgram()

# Define the decision variables
x = prog.NewContinuousVariables(2, "x")

# Define the objective function
prog.AddLinearCost(2 * x[0] + 3 * x[1])

# Define the constraints
prog.AddLinearConstraint(x[0] + 2 * x[1] <= 3)
prog.AddLinearConstraint(2 * x[0] + x[1] <= 3)
prog.AddLinearConstraint(x[0] >= 0)
prog.AddLinearConstraint(x[1] >= 0)

# Solve the linear program
result = Solve(prog)

# Check the solution result
assert result.is_success()

# Get the solution
x_sol = result.GetSolution(x)