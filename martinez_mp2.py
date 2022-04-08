# START

'''
    ES 26 Machine Problem 2

    Submitted by: Paula Joy B. Martinez
    Submitted on May 17, 2021
'''
from string import ascii_lowercase as letter # This code imports the lowercase alphabet letters that is invocable by the variable named "letter"

def budgetStatus(): 
    '''
    This function determines the user's budget status based on the computed residual amount of their disposable income after covering for expenses.
    The function returns either a budget surplus, deficit, or breakeven.
    '''
    computeBudgetStatus = budget_float["Disposable income:"] - budget_float["Expenses:"]
    status = ""
    if computeBudgetStatus == 0:
        status += "You have a breakeven!"
    elif computeBudgetStatus < 0:
        status += "You have a budget deficit!"
    elif computeBudgetStatus > 0:
        status += "You have a budget surplus!"
    else:
        status += "None"
    return status
    
# Computing for the cash ratio
def cashRatio():
    '''
    This function determines the financial ability of the user to cover their expenses using only cash and/or cash equivalents.
    A ratio of 1 means that for every one peso of expenses, the user has 1 peso in income to cover it.
    '''
    if budget_float["Disposable income:"] == 0:
        ratio = 0
    else:
        ratio = round(budget_float["Disposable income:"] / budget_float["Expenses:"], 2)
    return ratio


budget = {"Gross income:": 0, "Income tax:": 0, "Disposable income:": 0, "Expenses:": 0, "Emergency savings:": 0, "Investments:": 0}

# Scenario: The user would like to update their budget by adding components   
action0 = input("> Welcome to PennyWise! What would you like to do?\n\t").lower() # The .lower() method is for uniformity purposes of the input
if action0 == "update budget":
    addGrossIncome = input("> How much is your projected income?\n\t")
    while True:
        if addGrossIncome.isnumeric() == True: # The .isnumeric() method returns True if all the characters in a string are numbers.
            break
        else:
            print("\tPlease enter a positive number.")

    addIncomeTax = input("> How much is your projected income tax?\n\t")
    while True:
        if addIncomeTax.isnumeric() == True: # The .isnumeric() method returns True if all the characters in a string are numbers.
            break
        else:
            print("\tPlease enter a positive number.")
    
    addExpenses = input("> How much is your projected expenses?\n\t")
    while True:

        if addExpenses.isnumeric() == True and float(addExpenses) >= 0: # The .isnumeric() method returns True if all the characters in a string are numbers. Additionally, negative expenses are essentially income.
            break
        else:
            print("\tPlease enter a positive number.")

# Adding the input to the "budget" dict as values
    budget["Gross income:"] = addGrossIncome
    budget["Income tax:"] = addIncomeTax
    budget["Expenses:"] = addExpenses

# On the user's investments
    while True:
        addInvestments = input("> What percentage of your income would you like to invest? Express the allocation in percent. Enter 0 if you will not invest in the future.\n\t")
        if "%" in addInvestments:
            investmentAmount_deci = float(addInvestments[:len(addInvestments)-1])/100 # This removes the modulo/percent sign and converts the remaining character into float and divides it by 100 to get the portion of gross income to be invested
            if investmentAmount_deci > 0:
# Return on investments - Getting the potential rate of return
                currentStockValue = round(float(input("> What is the current value of your stock investment?\n\t")), 2)
                futureStockValue = round(float(input("> What is the forecasted value of the stock after one year? This can also be an estimate.\n\t")), 2)
                stockReturns = round(((futureStockValue - currentStockValue) / currentStockValue * 100), 2)
# Add the potential returns to the budget dict with floating values
                budget.update({"Potential investment returns:" : stockReturns}) # The .update() method adds a new key-value pair to the budget_float dictionary
                break
            else:
                print("Please enter a positive number.")
        elif addInvestments == "0":
            budget["Investments:"] = round(float(addInvestments), 2)
            break
        else:
            print("\tKindly enter a valid number. Express your allocation for investment in percentage terms.")
# On emergency savings
    while True:
        addSavings = input("> What percentage of your income would you like to save? Express the allocation in percent. Enter 0 if you will not save in the future.\n\t")
        if "%" in addSavings:
            savings_deci = float(addSavings[:len(addSavings)-1])/100 # This removes the modulo/percent sign and converts the remaining character into float and divides it by 100 to get the portion of gross income to be saved
            if savings_deci > 0:
                break
            else:
                print("Please enter a positive number.")
        elif addSavings == "0":
            budget["Emergency savings:"] = round(float(addSavings), 2)
            break
        else:
            print("\tKindly enter a valid number. Express your allocation for investment in percentage terms.")

# Converting the string values of the budget dict to floating values
    budget_float = {key:float(value) for key,value in budget.items()} # The .items() method views the elements in the dictionary as (key,value) pairs

# Add all the values to budget_float
# Computing for the disposable income, amount of investments, and savings
    budget_float["Disposable income:"] = budget_float["Gross income:"] - budget_float["Income tax:"]

# Adding the user's investments to the budget_float dictionary. Note that the 0 entry, has already been added to the budget_float dictionary in line 73 and line 82.
    if addInvestments != "0":
        budget_float["Investments:"] = round(((budget_float["Gross income:"]) * investmentAmount_deci), 2)
    if addSavings != "0":
        budget_float["Emergency savings:"] = round(((budget_float["Gross income:"]) * savings_deci), 2)

    print("\n\tPERSONAL BUDGET:")
    for key,value in budget_float.items():
        print("\n\t", key,value)
    print("\n\tBudget status:", budgetStatus())
    print("\tYour cash ratio is:", cashRatio(), end="")
    print(". You have", cashRatio(), "in income to cover each peso of your expenses.")
    

elif action0 == "check budget": # To check for the most recent entry after running the program
    print("\n\tPERSONAL BUDGET:")
    for key,value in budget_float.items():
        print("\t",key,value)
    print("\n\tBudget status:", budgetStatus())
    print("\tYour cash ratio is:", cashRatio(), end="")
    print(". You have", cashRatio(), "in income to cover each peso of your expenses.")
else:
    print("\tKindly choose if you want to update your budget or check your budget.")

