#### _Before creating a new test for an example script, please read the following information._

This folder contains a test suite for all of the examples in the `~/examples` directory. The examples are grouped by category (e.g., graphene, haldane, etc.) as in the `~/examples` directory, and each example has a folder of test files. When creating a new test for an example, please use the `make_test_example.py` script to generate a skeleton of the folder of test files. The way to use this script is as follows:

```bash
python make_test_example.py --group $NAMEOFGROUP --name $NAMEOFEXAMPLE
```

where the `$NAMEOFGROUP` is the name of the parent directory where the example lives, and `$NAMEOFEXAMPLE` is the example's name. This script will then create a folder structure `tests/test_example/$NAMEOFGROUP/$NAMEOFEXAMPLE` with the following content:

### `/golden_data` folder
The tests use 'golden data' from a previous working version of `pythtb` as a standard to compare to. This data is stored in the `golden_data` directory within each example's test folder. 

### `regen_golden_data.py` script
When releasing a new version, all of the golden data should be regenerated using the `regen_golden_data.py` file. Please double-check that the data is being saved as expected: with correct formatting and consistent file names across the test files.

### `run.py` script
This file contains a function `run()` which returns the relevant data being generated in the example script. This output is what will be compared against the golden data to ensure consistency across code updates.

### `test_$NAMEOFEXAMPLE.py` script
This file is what will be automatically detected and run by `pytest`. At the top of the file, the dictionary `OUTPUTS` must be modified to include the names of the golden data files. Be careful that the order in which the entries are added matches the order of what is returned in `run()`. If the data is something that is not savable by `NumPy`, such as pickled objects, be sure to modify the code as needed to make the appropriate comparisons. 
