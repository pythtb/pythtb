Before creating a new test for an example script, please read the following information.

This folder contains a test suite for all of the examples in the `~/examples` directory. The examples are grouped by category (e.g. graphene, haldane, etc.) as in the `~/examples` directory, and each example has a folder of test files. When creating a new example test, please use the `make_test_example.py` to generate a skeleton of the folder of test files.

```bash
python make_test_example.py --group $NAMEOFGROUP --name $NAMEOFEXAMPLE
```
This will create a folder under `$NAMEOFGROUP/$NAMEOFEXAMPLE` with the following:

### `/golden_data`
The tests use 'golden data' from a previous working version of `pythtb` as a standard to compare to. This data is stored in a `golden_data` directory within each example's test folder. 

### `regen_golden_data.py`
When releasing a new version, all of the golden data should be regenerated using the `regen_golden_data.py` file. Please double check that the data is being saved as expected: with correct formating, and consistent file names across the test files.

### `run.py`
This file contains a function `run()` which returns the relevant data being generated in the example script. This output is what will be compared against the golden data to ensure consistency across code updates.

### `test.py`
This file is what will be automatically detected and ran by `pytest`. If `run()` returns more than one output, or something that is not save-able by NumPy, be sure to modify the code where the assertion `np.allclose()` is made to something appropriate for your data type.

After each run of `pytest`, the pass/fail status will be written to `status.json` in the group directory. To see the status of all of the examples, run
```bash
python report_test_status.py
```
in the terminal. 