# CSC561 Interactive Decision Threshold
___

## Description of Activity:

This is a revision activity for CSC 561 at Andover, where I have built an interactive decision threshold tool for displaying how ML models, specifically a logistic regression, change the number of true positives/negatives and false positives/negatives as you adjust the decision boundary.

BTW: This uses only python files no notebooks, because notebooks are worse for github, modularity, and are just messier.

## Project Structure:
`app.py` — This file is for displaying the streamlit/web interface of the project
`log_reg.py` — This file contains the functions and data preparation for the logistic regression model.

## Run + Contribution Guide:

### Contributing:
1. Start by cloning the repo. In your terminal write: `git clone https://github.com/MilanSuri/CSC561InteractiveDecisionThreshold.git`
2. Then cd to that directory. In terminal write: `cd CSC561InteractiveDecisionThreshold`
3. To ensure you're on the latest version run `git pull` in terminal.
4. Once you have an up to date local copy, create a separate branch by running: `git checkout -b {name}`
5. Then confirm you're not on the main branch by running `git branch`. You should see an asterisk by the current branch.
6. If you're not on your branch change to it by doing `git switch {branch_name}`
7. Then once you've confirmed you're on the right branch, make your modifications to `app.py` or `log_reg.py`
8. Once your changes are complete run `git status` and `git diff` to see the changes you've made overall.
9. Then do `git add` to add the modified files. Example: `git add app.py`.
10. Once you're ready, commit the change and add a clear, descriptive message of what you did. Do this through `git commit -m {commit msg}`
11. Then once you want to push to this codebase you run: `git push -u origin {branch_name}`
12. Then go to github and you should be prompted to submit a pull request (PR)
13. When submitting a PR, write a detailed summary of what you did, try to avoid AI as more open source ppl have become skeptical
14. A person who has auth to merge will review, provide comments, merge, or reject the PR.


### Running:
1. You can install UV (dependency manager) through:
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh | iex"`
   - Homebrew: `brew install uv`
   - Pip: `pip install uv`
2. We use UV for this project so you can install UV and run the command: `uv sync`.
3. After adding the dependencies to your env, you can run: `streamlit run app.py` to see the streamlit app.

## Resources Used:
- https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
- https://scikit-learn.org/stable/
- https://matplotlib.org/stable/index.html
