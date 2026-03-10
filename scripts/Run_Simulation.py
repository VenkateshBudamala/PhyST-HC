# Run_simulation.py

exec(open('Inputs.py').read())

configs = {
    'Q': 'Functions_Q.py',
    'POC': 'Functions_POC.py',
    'DOC': 'Functions_DOC.py'
}

for variable, func_file in configs.items():

    print("\n")
    print("╔" + "═"*78 + "╗")
    print(f"║{'RUNNING VARIABLE: ' + variable:^78}║")
    print("╚" + "═"*78 + "╝")
    print("\n")

    exec(open(func_file).read())
    exec(open('Main_Module.py').read())

